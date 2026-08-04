import torch
import math
import torch.nn.functional as F

from napl.sim.base import napl_base
from loguru import logger
# Operation imports stay inside __init__ to avoid the module-operation import cycle.
from napl.sim.module._shared import _init_mgu_params


class mgu_hardfxp(napl_base):
    r"""Apply a quantization-aware MGU cell with hard range bounds.

    Use this single-shot cell to train or evaluate an MGU while rounding operands
    to the configured fixed-point format. ``round_fxp`` supplies the
    straight-through gradient.

    The precise target is the Minimal Gated Unit recurrence

    .. math::

       f = \sigma\!\left(W_f [h, x] + b_f\right),\qquad
       n = \tanh\!\left(W_n [f \odot h, x] + b_n\right),

    .. math::

       h' = (1 - f) \odot n + f \odot h.

    Let :math:`Q(\cdot)` be the ``round_fxp`` quantizer for the configured
    **intwidth** and **fracwidth**. The cell evaluates the same recurrence with
    hard activations and applies :math:`Q` to every operand and intermediate
    result,

    .. math::

       f = \sigma_h\!\left(Q\!\left(\mathrm{clamp}\left(
       W'_f\,[Q(h), Q(x)]' + b'_f,\, -1,\, 1\right)\right)\right),\qquad
       n = \tanh_h\!\left(Q\!\left(W'_n\,[Q(f \odot h), Q(x)]' + b'_n\right)\right),

    .. math::

       h' = \mathrm{clamp}\!\left(
       Q(n) - Q\!\left(Q(f) \odot Q(n)\right) + Q(f \odot h),\, -1,\, 1\right),

    where a primed symbol denotes its quantized form,
    :math:`\sigma_h(v) = \mathrm{clip}(v/2 + 1/2, 0, 1)`, and
    :math:`\tanh_h(v) = \mathrm{clamp}(v, -1, 1)`. The quantization step, the
    hard activations, and the output clamp are the departures from the target.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu_hardfxp

        cell = mgu_hardfxp(2, 3, config={"hard": True,
                                        "intwidth": 3, "fracwidth": 4})
        hidden = cell(torch.zeros(1, 2))
    """
    #: Whether calls process one stream timestep; this cell is single-shot.
    streaming = False


    def __init__(self, input_size, hidden_size, bias=True, config={'hard': True, 'intwidth': 3, 'fracwidth': 4}):
        """Construct the fixed-point MGU and initialize trainable parameters.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden features.
            bias: Create trainable gate biases when ``True``. Defaults to ``True``.
            config: Configuration mapping with these keys:

                * **hard** - Use hard gate activations when ``True``; otherwise
                  use ``Sigmoid`` and ``Tanh``. Defaults to ``True``.
                * **intwidth** - Integer-bit count passed to ``round_fxp``.
                  Defaults to ``3``.
                * **fracwidth** - Fractional-bit count passed to ``round_fxp``.
                  Defaults to ``4``.
                * **name** - Optional instance label. Defaults to ``None``.
        """
        super().__init__(config, [])
        #: Number of features in each input vector.
        self.input_size = input_size
        #: Number of features in each hidden-state vector.
        self.hidden_size = hidden_size
        #: Whether the forget and new gates include trainable biases.
        self.bias = bias
        #: Whether the forget and new gates use hard activations.
        self.hard = config.get('hard', True)
        from napl.sim.operation import sigmoid_hub, tanh_hub, round_fxp
        #: Hard-tanh operator that bounds intermediate and output values.
        self.htanh = tanh_hub()
        #: Fixed-point quantizer applied to recurrent operands and parameters.
        self.trunc = round_fxp({'intwidth': config.get('intwidth', 3), 'fracwidth': config.get('fracwidth', 4)})
        #: Activation applied to the forget gate.
        self.fg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        #: Activation applied to the candidate hidden state.
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()
        _init_mgu_params(self, input_size, hidden_size, bias)

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset local recurrent state.

        The cell stores no hidden state between calls, so this hook returns
        ``None`` without changing parameters or quantization settings.
        """
        pass


    def forward(self, input, hx=None):
        """Compute one quantized MGU recurrence.

        Args:
            input: Tensor of shape ``(batch, input_size)``.
            hx: Optional previous hidden tensor of shape
                ``(batch, hidden_size)``. Defaults to zeros.

        Returns:
            The next hidden tensor of shape ``(batch, hidden_size)``, bounded to
            ``[-1, 1]``.

        Operands are quantized during the call. The cell does not store ``hx`` or
        advance ``timestep_cur``.
        """
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        t = self.trunc
        t_hx, t_input = t(hx), t(input)
        fg_ug_in = torch.cat((t_hx, t_input), 1)
        fg_in = self.htanh(F.linear(t(fg_ug_in), t(self.weight_f), t(self.bias_f)))
        fg = self.fg_sigmoid(t(fg_in))
        t_fg = t(fg)
        fg_hx = t_fg * t_hx
        t_fg_hx = t(fg_hx)
        ng_ug_in = torch.cat((t_fg_hx, t_input), 1)
        ng = self.ng_tanh(t(F.linear(t(ng_ug_in), t(self.weight_n), t(self.bias_n))))
        t_ng = t(ng)
        fg_ng = t_fg * t_ng
        return self.htanh(t_ng - t(fg_ng) + t_fg_hx)

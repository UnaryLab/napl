import torch
import math
import torch.nn.functional as F

from napl.sim.base import napl_base
from loguru import logger
# Operation imports stay inside __init__ to avoid the module-operation import cycle.
from napl.sim.module._shared import _init_mgu_params


class mgu_hard(napl_base):
    r"""Apply a trainable single-shot MGU cell with bounded hard activations.

    Use this binary-domain cell when intermediate and output values must remain in
    the legal unary range. It uses hard sigmoid and hard tanh by default and does
    not advance the streaming timestep.

    The precise target is the Minimal Gated Unit recurrence

    .. math::

       f = \sigma\!\left(W_f [h, x] + b_f\right),\qquad
       n = \tanh\!\left(W_n [f \odot h, x] + b_n\right),

    .. math::

       h' = (1 - f) \odot n + f \odot h.

    The cell evaluates that recurrence with hard activations and an added clamp
    on the forget-gate linear and on the output, keeping every intermediate value
    in ``[-1, 1]``,

    .. math::

       f = \sigma_h\!\left(\mathrm{clamp}\left(
       W_f [h, x] + b_f,\, -1,\, 1\right)\right),\qquad
       n = \tanh_h\!\left(W_n [f \odot h, x] + b_n\right),

    .. math::

       h' = \mathrm{clamp}\!\left(n - f \odot n + f \odot h,\, -1,\, 1\right),

    where :math:`\sigma_h(v) = \mathrm{clip}(v/2 + 1/2, 0, 1)` and
    :math:`\tanh_h(v) = \mathrm{clamp}(v, -1, 1)` when **hard** is ``True``, and
    the exact ``Sigmoid`` and ``Tanh`` otherwise. The hard activations and the
    clamps are the only departures from the target.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu_hard

        cell = mgu_hard(2, 3)
        hidden = cell(torch.zeros(1, 2))

    References
    ----------
    *Simplified Minimal Gated Unit Variations for RNNs*.
    """
    #: Whether calls process one stream timestep; this cell is single-shot.
    streaming = False


    def __init__(self, input_size, hidden_size, bias=True, config={'hard': True}):
        """Construct the MGU cell and initialize its trainable parameters.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden features.
            bias: Create trainable forget- and new-gate biases when ``True``.
                Defaults to ``True``.
            config: Configuration mapping with **hard**. ``True`` selects hard
                sigmoid and hard tanh; ``False`` selects ``Sigmoid`` and ``Tanh``
                for the gates while retaining the explicit bounding clamps.
                Defaults to ``True``. **name** is an optional instance label and
                defaults to ``None``.
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
        from napl.sim.operation import sigmoid_hub, tanh_hub
        #: Hard-tanh operator that bounds intermediate and output values.
        self.htanh = tanh_hub()
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
        ``None`` without changing its trainable parameters.
        """
        pass


    def forward(self, input, hx=None):
        """Compute one MGU recurrence in the binary domain.

        Args:
            input: Tensor of shape ``(batch, input_size)``.
            hx: Optional previous hidden tensor of shape
                ``(batch, hidden_size)``. Defaults to zeros.

        Returns:
            The next hidden tensor of shape ``(batch, hidden_size)``, bounded to
            ``[-1, 1]``.

        The call does not store ``hx`` or change ``timestep_cur``. Gradients flow
        to the input and trainable parameters.
        """
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        fg_ug_in = torch.cat((hx, input), 1)
        fg_in = self.htanh(F.linear(fg_ug_in, self.weight_f, self.bias_f))
        fg = self.fg_sigmoid(fg_in)
        fg_hx = fg * hx
        ng_ug_in = torch.cat((fg_hx, input), 1)
        ng = self.ng_tanh(F.linear(ng_ug_in, self.weight_n, self.bias_n))
        fg_ng = fg * ng
        return self.htanh(ng - fg_ng + fg_hx)

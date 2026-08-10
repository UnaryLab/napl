import torch
import torch.nn.functional as F

from napl.sim.base import napl_base
from napl.sim.module._shared import _init_mgu_params
from .round_fxp import round_fxp
from napl.sim.operation import sigmoid_fxp, tanh_fxp


# Single source for every optional key: the signature default and the per-key fallback.
_DEFAULT_CONFIG = {
    'hard': True,
    'intwidth': 3,
    'fracwidth': 4,
}


class mgu_hard_fxp(napl_base):
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

    The cell evaluates that recurrence with the hard activations

    .. math::

       \sigma_h(v) = \mathrm{clip}\!\left(\frac{v}{2} + \frac{1}{2},\, 0,\, 1
       \right),\qquad
       \tanh_h(v) = \mathrm{clamp}(v, -1, 1)

    when **hard** is ``True``, and with ``Sigmoid`` and ``Tanh`` otherwise. Every
    operand, parameter, and intermediate result is rounded to the fixed-point
    format given by **intwidth** and **fracwidth** before it enters the next
    stage; the returned hidden state carries no rounding after its final clamp.
    The forget-gate linear and the returned hidden state are clamped to
    ``[-1, 1]``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu_hard_fxp

        cell = mgu_hard_fxp(2, 3, config={"hard": True,
                                          "intwidth": 3, "fracwidth": 4})
        hidden = cell(torch.zeros(1, 2))

    .. container:: api-references

        .. rubric:: References

        *Simplified minimal gated unit variations for recurrent neural networks*, MWSCAS, 2017.
    """
    #: Whether calls process one stream timestep; this cell is single-shot.
    streaming = False


    def __init__(self, input_size, hidden_size, bias=True, config=_DEFAULT_CONFIG):
        """Construct the fixed-point MGU and initialize trainable parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **input_size** – Number of input features.
            - **hidden_size** – Number of hidden features.
            - **bias** – Create trainable gate biases when ``True``; the default is ``True``.
            - **config** – Configuration mapping. Omitted keys fall back to the same defaults.

              - **hard**: Use hard gate activations when ``True``, or ``Sigmoid`` and ``Tanh`` when ``False``; the default is ``True``.
              - **intwidth**: Integer-bit count passed to ``round_fxp``; the default is ``3``.
              - **fracwidth**: Fractional-bit count passed to ``round_fxp``; the default is ``4``.
              - **name**: Optional instance label.
        """
        super().__init__(config, [], optional_key_list=list(_DEFAULT_CONFIG))
        cfg = {**_DEFAULT_CONFIG, **config}
        #: Number of features in each input vector.
        self.input_size = input_size
        #: Number of features in each hidden-state vector.
        self.hidden_size = hidden_size
        #: Whether the forget and new gates include trainable biases.
        self.bias = bias
        #: Whether the forget and new gates use hard activations.
        self.hard = cfg['hard']
        #: Hard-tanh operator that bounds intermediate and output values.
        self.htanh = tanh_fxp()
        #: Fixed-point quantizer applied to recurrent operands and parameters.
        self.trunc = round_fxp({'intwidth': cfg['intwidth'], 'fracwidth': cfg['fracwidth']})
        #: Activation applied to the forget gate.
        self.fg_sigmoid = sigmoid_fxp() if self.hard else torch.nn.Sigmoid()
        #: Activation applied to the candidate hidden state.
        self.ng_tanh = tanh_fxp() if self.hard else torch.nn.Tanh()
        _init_mgu_params(self, input_size, hidden_size, bias)


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

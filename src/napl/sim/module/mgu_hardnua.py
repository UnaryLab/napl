import torch
import torch.nn.functional as F

from napl.sim.base import napl_base
from napl.sim.module._shared import _init_mgu_params
from napl.sim.operation import sigmoid_hub, tanh_hub


# Single source for every optional key: the signature default and the per-key fallback.
_DEFAULT_CONFIG = {
    'hard': True,
}


class mgu_hardnua(napl_base):
    r"""Apply a trainable MGU without unary-range clamps around linear stages.

    Use this single-shot cell to match the non-unary-aware UnarySim variant or to
    study the effect of removing the range clamps from :class:`mgu_hard`. The
    forget-gate linear input and output omit the hard-tanh clamps, so
    intermediate values and ``hy`` may leave the legal unary range. The cell is
    single-shot and trainable.

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

    when **hard** is ``True``, and with ``Sigmoid`` and ``Tanh`` otherwise. Apart
    from the bound that :math:`\tanh_h` places on :math:`n`, no stage carries a
    range clamp, so :math:`h'` may leave ``[-1, 1]``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu_hardnua

        cell = mgu_hardnua(2, 3)
        hidden = cell(torch.zeros(1, 2))

    .. container:: api-references

        .. rubric:: References

        *Simplified minimal gated unit variations for recurrent neural networks*, MWSCAS, 2017.
    """
    #: Whether calls process one stream timestep; this cell is single-shot.
    streaming = False


    def __init__(self, input_size, hidden_size, bias=True, config=_DEFAULT_CONFIG):
        """Construct the non-unary-aware cell and initialize its parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **input_size** – Number of input features.
            - **hidden_size** – Number of hidden features.
            - **bias** – Create trainable gate biases when ``True``; the default is ``True``.
            - **config** – Configuration mapping. Omitted keys fall back to the same defaults.

              - **hard**: Use hard sigmoid and hard tanh when ``True``, or ``Sigmoid`` and ``Tanh`` when ``False``; the default is ``True``.
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
        ``None`` without changing trainable parameters.
        """
        pass


    def forward(self, input, hx=None):
        """Compute one unclamped MGU recurrence.

        Args:
            input: Tensor shaped ``(batch, input_size)``.
            hx: Optional previous hidden tensor shaped
                ``(batch, hidden_size)``. Defaults to zeros.

        Returns:
            Next hidden tensor shaped ``(batch, hidden_size)``. Values may leave
            ``[-1, 1]``.

        The call does not store ``hx`` or advance ``timestep_cur``.
        """
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        fg = self.fg_sigmoid(F.linear(torch.cat((hx, input), 1), self.weight_f, self.bias_f))
        fg_hx = fg * hx
        ng = self.ng_tanh(F.linear(torch.cat((fg_hx, input), 1), self.weight_n, self.bias_n))
        return ng - fg * ng + fg_hx

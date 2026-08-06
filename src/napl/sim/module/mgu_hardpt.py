import torch
import torch.nn.functional as F

from napl.sim.base import napl_base
from napl.sim.operation import sigmoid_hub, tanh_hub
from napl.sim.module._shared import _init_mgu_pt_params


# Single source for every optional key: the signature default and the per-key fallback.
_DEFAULT_CONFIG = {
    'hard': True,
}


class mgu_hardpt(napl_base):
    r"""Apply a PyTorch-layout MGU cell with bounded hard activations.

    Use this single-shot cell when parameters must follow the two-chunk
    input-hidden and hidden-hidden layout used by PyTorch recurrent cells. It is
    a binary-domain Minimal Gated Unit with separate input-hidden and
    hidden-hidden linears split into forget and new gates. Hard sigmoid, hard
    tanh, and explicit range clamps keep intermediate values in the legal unary
    range. The cell is single-shot and trainable.

    The precise target is the Minimal Gated Unit recurrence in the PyTorch
    two-chunk layout, with :math:`g^i = W_{ih} x + b_{ih}` and
    :math:`g^h = W_{hh} h + b_{hh}` each split into forget and new halves,

    .. math::

       f = \sigma\!\left(g^i_f + g^h_f\right),\qquad
       n = \tanh\!\left(g^i_n + f \odot g^h_n\right),\qquad
       h' = (1 - f) \odot n + f \odot h.

    The cell evaluates that recurrence with the hard activations

    .. math::

       \sigma_h(v) = \mathrm{clip}\!\left(\frac{v}{2} + \frac{1}{2},\, 0,\, 1
       \right),\qquad
       \tanh_h(v) = \mathrm{clamp}(v, -1, 1)

    when **hard** is ``True``, and with ``Sigmoid`` and ``Tanh`` otherwise. Both
    gate linears, the forget-gate sum, and the returned hidden state are clamped
    to ``[-1, 1]``, and those clamps stay hard regardless of that setting.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu_hardpt

        cell = mgu_hardpt(2, 3)
        hidden = cell(torch.zeros(1, 2))

    .. container:: api-references

        .. rubric:: References

        *Simplified minimal gated unit variations for recurrent neural networks*, MWSCAS, 2017.
    """
    #: Whether calls process one stream timestep; this cell is single-shot.
    streaming = False


    def __init__(self, input_size, hidden_size, bias=True, config=_DEFAULT_CONFIG):
        """Construct the PyTorch-layout cell and initialize its parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **input_size** – Number of input features.
            - **hidden_size** – Number of hidden features.
            - **bias** – Create input-hidden and hidden-hidden biases when ``True``; the default is ``True``.
            - **config** – Configuration mapping. Omitted keys fall back to the same defaults.

              - **hard**: Use hard gate activations when ``True``, or ``Sigmoid`` and ``Tanh`` when ``False``, with the range clamps always active; the default is ``True``.
              - **name**: Optional instance label.
        """
        super().__init__(config, [], optional_key_list=list(_DEFAULT_CONFIG))
        cfg = {**_DEFAULT_CONFIG, **config}
        #: Number of features in each input vector.
        self.input_size = input_size
        #: Number of features in each hidden-state vector.
        self.hidden_size = hidden_size
        #: Whether the cell includes trainable input and hidden biases.
        self.bias = bias
        #: Whether the forget and new gates use hard activations.
        self.hard = cfg['hard']
        #: Activation applied to the forget gate.
        self.fg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        #: Activation applied to the candidate hidden state.
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()
        #: Hard-tanh operator that bounds intermediate and output values.
        self.htanh = tanh_hub()
        # PyTorch's two-chunk parameter layout stores forget rows before new rows.
        _init_mgu_pt_params(self, input_size, hidden_size, bias, 2)
        #: Trainable input-hidden weight shaped ``(2 * hidden_size, input_size)``, with the forget rows first.
        self.weight_ih: torch.nn.Parameter
        #: Trainable hidden-hidden weight shaped ``(2 * hidden_size, hidden_size)``, with the forget rows first.
        self.weight_hh: torch.nn.Parameter
        #: Trainable input-hidden bias shaped ``(2 * hidden_size,)``, or ``None`` when **bias** is ``False``.
        self.bias_ih: torch.nn.Parameter
        #: Trainable hidden-hidden bias shaped ``(2 * hidden_size,)``, or ``None`` when **bias** is ``False``.
        self.bias_hh: torch.nn.Parameter

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
        """Compute one bounded MGU recurrence.

        Args:
            input: Tensor shaped ``(batch, input_size)``.
            hx: Optional previous hidden tensor shaped
                ``(batch, hidden_size)``. Defaults to zeros.

        Returns:
            Next hidden tensor shaped ``(batch, hidden_size)``, bounded to
            ``[-1, 1]``.

        The call does not store ``hx`` or advance ``timestep_cur``.
        """
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        # Range clamps stay hard regardless of the configured gate activations.
        gate_i = self.htanh(F.linear(input, self.weight_ih, self.bias_ih))
        gate_h = self.htanh(F.linear(hx, self.weight_hh, self.bias_hh))
        i_f, i_n = gate_i.chunk(2, 1)
        h_f, h_n = gate_h.chunk(2, 1)
        fg = self.fg_sigmoid(self.htanh(i_f + h_f))
        ng = self.ng_tanh(i_n + fg * h_n)
        return self.htanh(ng - fg * ng + fg * hx)

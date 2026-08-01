import torch
import math
import torch.nn.functional as F

from napl.utils import *
from napl.sim.base import napl_base

# Operation imports stay inside __init__ to avoid the module-operation import cycle.


class mgu_hardpt(napl_base):
    """Apply a PyTorch-layout MGU cell with bounded hard activations.

    Use this single-shot cell when parameters must follow the two-chunk
    input-hidden and hidden-hidden layout used by PyTorch recurrent cells. It is
    a binary-domain Minimal Gated Unit with separate input-hidden and
    hidden-hidden linears split into forget and new gates. Hard sigmoid, hard
    tanh, and explicit range clamps keep intermediate values in the legal unary
    range. The cell is single-shot and trainable.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu_hardpt

        cell = mgu_hardpt(2, 3)
        hidden = cell(torch.zeros(1, 2))

    References
    ----------
    *Simplified Minimal Gated Unit Variations for RNNs*.
    """
    #: Whether calls process one stream timestep; this cell is single-shot.
    streaming = False
    def __init__(self, input_size, hidden_size, bias=True, config={'hard': True}):
        """Construct the PyTorch-layout cell and initialize its parameters.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden features.
            bias: Create input-hidden and hidden-hidden biases when ``True``.
                Defaults to ``True``.
            config: Configuration mapping with **hard**. ``True`` uses hard gate
                activations; ``False`` uses ``Sigmoid`` and ``Tanh`` for the gates.
                Explicit hard-tanh range clamps remain active. Defaults to ``True``.
                **name** is an optional instance label and defaults to ``None``.
        """
        super().__init__(config, [])
        #: Number of features in each input vector.
        self.input_size = input_size
        #: Number of features in each hidden-state vector.
        self.hidden_size = hidden_size
        #: Whether the cell includes trainable input and hidden biases.
        self.bias = bias
        #: Whether the forget and new gates use hard activations.
        self.hard = config.get('hard', True)
        from napl.sim.operation import sigmoid_hub, tanh_hub
        #: Activation applied to the forget gate.
        self.fg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        #: Activation applied to the candidate hidden state.
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()
        # PyTorch's two-chunk parameter layout stores forget rows before new rows.
        #: Trainable input-to-gate weights in forget-then-new order.
        self.weight_ih = torch.nn.Parameter(torch.empty(2 * hidden_size, input_size))
        #: Trainable hidden-to-gate weights in forget-then-new order.
        self.weight_hh = torch.nn.Parameter(torch.empty(2 * hidden_size, hidden_size))
        if bias:
            #: Trainable input-to-gate bias in forget-then-new order.
            self.bias_ih = torch.nn.Parameter(torch.empty(2 * hidden_size))
            #: Trainable hidden-to-gate bias in forget-then-new order.
            self.bias_hh = torch.nn.Parameter(torch.empty(2 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        stdv = 1.0 / math.sqrt(hidden_size)
        for w in [self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh]:
            if w is not None:
                w.data = truncated_normal(w, 0.0, stdv)

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
        gate_i = F.hardtanh(F.linear(input, self.weight_ih, self.bias_ih), -1.0, 1.0)
        gate_h = F.hardtanh(F.linear(hx, self.weight_hh, self.bias_hh), -1.0, 1.0)
        i_f, i_n = gate_i.chunk(2, 1)
        h_f, h_n = gate_h.chunk(2, 1)
        fg = self.fg_sigmoid(F.hardtanh(i_f + h_f, -1.0, 1.0))
        ng = self.ng_tanh(i_n + fg * h_n)
        return F.hardtanh(ng - fg * ng + fg * hx, -1.0, 1.0)

import torch
import math
import torch.nn.functional as F

from napl.utils import *
from napl.sim.base import napl_base

# NB: no operation imports needed; the hard activations (sigmoid_hub/tanh_hub) are
# imported lazily inside __init__ to respect the module<->operation import cycle.


class gru_hardnuapt(napl_base):
    """Apply a PyTorch-layout GRU cell with optional hard activations.

    Use this single-shot cell to match UnarySim ``HardGRUCellNUAPT`` while keeping
    the standard three-chunk ``GRUCell`` parameter layout. It is a standard GRU cell in the binary (float) domain with hard activations: sigmoid ->
    scaled hard sigmoid, tanh -> hard tanh. Not fully unary-aware (NUA): intermediate
    values are not bounded to the legal unary range. PyTorch GRUCell equations and
    weight layout (3-chunk weight_ih/weight_hh). Single-shot; trainable.
    Port of UnarySim ``HardGRUCellNUAPT``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import gru_hardnuapt

        cell = gru_hardnuapt(2, 3)
        hidden = cell(torch.zeros(1, 2))
    """
    streaming = False
    def __init__(self, input_size, hidden_size, bias=True, config={'hard': True}):
        """Construct the GRU cell and initialize trainable parameters.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden features.
            bias: Create input-hidden and hidden-hidden biases when ``True``.
                Defaults to ``True``.
            config: Configuration mapping with **hard**. ``True`` uses scaled hard
                sigmoid and hard tanh; ``False`` uses ``Sigmoid`` and ``Tanh``.
                Defaults to ``True``. **name** is an optional instance label and
                defaults to ``None``.
        """
        super().__init__(config, [])
        self.input_size, self.hidden_size, self.bias = input_size, hidden_size, bias
        self.hard = config.get('hard', True)
        from napl.sim.operation import sigmoid_hub, tanh_hub
        self.rg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        self.ug_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()

        self.weight_ih = torch.nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = torch.nn.Parameter(torch.empty(3 * hidden_size))
            self.bias_hh = torch.nn.Parameter(torch.empty(3 * hidden_size))
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
        """Compute one GRU recurrence.

        Args:
            input: Tensor shaped ``(batch, input_size)``.
            hx: Optional previous hidden tensor shaped
                ``(batch, hidden_size)``. Defaults to zeros.

        Returns:
            Next hidden tensor shaped ``(batch, hidden_size)``. Because this
            variant is not unary-aware, values may leave ``[-1, 1]``.

        The call does not store ``hx`` or advance ``timestep_cur``.
        """
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        gate_i = F.linear(input, self.weight_ih, self.bias_ih)
        gate_h = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gate_i.chunk(3, 1)
        h_r, h_z, h_n = gate_h.chunk(3, 1)

        rg = self.rg_sigmoid(i_r + h_r)          # reset gate
        ug = self.ug_sigmoid(i_z + h_z)          # update gate
        ng = self.ng_tanh(i_n + rg * h_n)        # new gate
        # output: hy = (1-ug)*ng + ug*hx
        return (1 - ug) * ng + ug * hx

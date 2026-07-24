import torch
import math
import torch.nn.functional as F

from napl.utils import *
from napl.sim.base import napl_base

# NB: no operation imports needed; the hard activations (sigmoid_hub/tanh_hub) are
# imported lazily inside __init__ to respect the module<->operation import cycle.


class gru_hardnuapt(napl_base):
    """
    Standard GRU cell in the binary (float) domain with hard activations: sigmoid ->
    scaled hard sigmoid, tanh -> hard tanh. Not fully unary-aware (NUA): intermediate
    values are not bounded to the legal unary range. PyTorch GRUCell equations and
    weight layout (3-chunk weight_ih/weight_hh). Single-shot; trainable.
    Port of UnarySim HardGRUCellNUAPT.
    """
    streaming = False
    def __init__(self, input_size, hidden_size, bias=True, config={'hard': True}):
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

    def forward(self, input, hx=None):
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

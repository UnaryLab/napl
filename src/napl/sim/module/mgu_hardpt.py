import torch
import math
import torch.nn.functional as F

from napl.utils import *
from napl.sim.base import napl_base

# NB: operation primitives are imported lazily inside __init__ (not at module top),
# because napl.sim.operation.mul_csg imports napl.sim.module.encoder, so a top-level import here
# creates a circular import when napl.sim.operation is loaded before napl.sim.module.


class mgu_hardpt(napl_base):
    """
    Minimal Gated Unit (MGU) cell in the binary (float) domain, PyTorch RNNCell style
    (separate input-hidden / hidden-hidden gate linears, chunked into forget/new gates)
    with hard activations: sigmoid -> hard sigmoid, tanh -> hard tanh, and explicit
    hard-tanh clamps so every intermediate value stays in the legal unary range.
    Single-shot; trainable. Port of UnarySim HardMGUCellPT.
    Refs: "Simplified Minimal Gated Unit Variations for RNNs".
    """
    streaming = False
    def __init__(self, input_size, hidden_size, bias=True, config={'hard': True}):
        super().__init__(config, [])
        self.input_size, self.hidden_size, self.bias = input_size, hidden_size, bias
        self.hard = config.get('hard', True)
        from napl.sim.operation import sigmoid_hub, tanh_hub
        self.fg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()
        # PT (RNNCellBase, num_chunks=2) parameter layout: rows [forget; new]
        self.weight_ih = torch.nn.Parameter(torch.empty(2 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.empty(2 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = torch.nn.Parameter(torch.empty(2 * hidden_size))
            self.bias_hh = torch.nn.Parameter(torch.empty(2 * hidden_size))
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
        # the explicit hard-tanh clamps (always hard): F.hardtanh directly, same math
        # as operation.tanh_hub but without the nn.Module dispatch on the hot path
        gate_i = F.hardtanh(F.linear(input, self.weight_ih, self.bias_ih), -1.0, 1.0)
        gate_h = F.hardtanh(F.linear(hx, self.weight_hh, self.bias_hh), -1.0, 1.0)
        i_f, i_n = gate_i.chunk(2, 1)
        h_f, h_n = gate_h.chunk(2, 1)
        # forget gate
        fg = self.fg_sigmoid(F.hardtanh(i_f + h_f, -1.0, 1.0))
        # new gate: ng = tanh(i_n + fg * h_n)
        ng = self.ng_tanh(i_n + fg * h_n)
        # output: hy = hardtanh((1 - fg) * ng + fg * hx)
        return F.hardtanh(ng - fg * ng + fg * hx, -1.0, 1.0)

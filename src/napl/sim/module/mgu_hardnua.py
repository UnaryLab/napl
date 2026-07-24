import torch
import torch.nn.functional as F

from napl.sim.base import napl_base
from napl.sim.module.rnn import _init_mgu_params

# NB: operation primitives are imported lazily inside __init__ (not at module top);
# see napl.sim.module.rnn for the module<->operation import-cycle rationale.


class mgu_hardnua(napl_base):
    """
    Non-unary-aware (NUA) hard MGU cell: mgu_hard without the hard-tanh clamps on the
    forget-gate linear input and on the output, so intermediates and hy may leave the
    legal unary range. Single-shot; trainable.
    Port of UnarySim HardMGUCellNUA.
    """
    streaming = False
    def __init__(self, input_size, hidden_size, bias=True, config={'hard': True}):
        super().__init__(config, [])
        self.input_size, self.hidden_size, self.bias = input_size, hidden_size, bias
        self.hard = config.get('hard', True)
        from napl.sim.operation import sigmoid_hub, tanh_hub
        self.fg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()
        _init_mgu_params(self, input_size, hidden_size, bias)

    def forward(self, input, hx=None):
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        fg = self.fg_sigmoid(F.linear(torch.cat((hx, input), 1), self.weight_f, self.bias_f))
        fg_hx = fg * hx
        ng = self.ng_tanh(F.linear(torch.cat((fg_hx, input), 1), self.weight_n, self.bias_n))
        # hy = ng - fg*ng + fg*hx, unclamped (the NUA difference vs mgu_hard)
        return ng - fg * ng + fg_hx

import torch

from napl.sim.base import napl_base
from napl.utils import pow2_lshift, pow2_rshift

class _round_ste_fn(torch.autograd.Function):
    """
    Straight-through rounding: round to a fixed-point grid on the forward pass, pass
    the gradient through unchanged on the backward pass. Plain round()/floor()/ceil()
    have zero gradient everywhere, which is useless for quantization-aware training.
    Semantics: round(x << f).clamp(min,max) >> f, using the float-safe pow2 shift shims
    instead of integer operators.
    """
    @staticmethod
    def forward(ctx, input, fracwidth, min_val, max_val):
        # pow2_lshift returns a fresh tensor (never aliases input), so round/clamp it
        # in place to avoid two intermediate allocations; values are identical.
        scaled = pow2_lshift(input, fracwidth)
        scaled.round_().clamp_(min_val, max_val)
        return pow2_rshift(scaled, fracwidth)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

def round_ste(input, fracwidth=0, min_val=None, max_val=None):
    """
    Straight-through-estimator round of a float tensor to a fixed-point grid with
    `fracwidth` fractional bits, optionally clamped to [min_val, max_val]. The gradient
    passes through unchanged, so it is differentiable for quant-aware training.
    """
    if min_val is None:
        min_val = float('-inf')
    if max_val is None:
        max_val = float('inf')
    input_float = input if input.dtype == torch.float32 else input.to(torch.float32)
    output = _round_ste_fn.apply(input_float, fracwidth, min_val, max_val)
    return output if input.dtype == torch.float32 else output.to(input.dtype)

class round_fxp(napl_base):
    """
    Quantize data to signed fixed-point format (sign, intwidth, fracwidth): the value
    is rounded onto a 2**fracwidth grid and clamped to
    [1 - 2**(intwidth+fracwidth), 2**(intwidth+fracwidth) - 1] / 2**fracwidth, using
    straight-through rounding so it stays differentiable. Single-shot, binary-domain.
    """
    streaming = False
    def __init__(
            self,
            config={
                'intwidth': 3,
                'fracwidth': 4,
            }
        ):
        super().__init__(config, ['intwidth', 'fracwidth'])

        self.intwidth = config['intwidth']
        self.fracwidth = config['fracwidth']
        self.max_val = 2**(self.intwidth + self.fracwidth) - 1
        self.min_val = 1 - 2**(self.intwidth + self.fracwidth)
        # RTL latency: the round_fxp clamp (src/napl/imp/operation/round_fxp)
        # is purely combinational (saturating clamp on the fixed-point code).
        self.delay = 0

    def forward(self, input):

        if input is None:
            return None
        return round_ste(input, self.fracwidth, self.min_val, self.max_val)

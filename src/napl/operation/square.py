import torch

from napl.utils import *
from napl.base import napl_base, hw_params
from napl.operation import dff


class square_dff(napl_base):
    """
    This module is for unary square with AND gate and dff, supporting unipolar/bipolar.
    References:
    1) uGEMM: Unary Computing Architecture for GEMM Applications
    2) uGEMM: Unary Computing for GEMM Applications
    """
    def __init__(
            self,
            config={
                'polarity': 'bipolar',
                'depth': 1,
            }
        ):
        super().__init__(config, ['polarity'], polarity_required=True)
        self.hw = hw_params(pp_delay=0)

        # the depth of input
        self.dff = dff(config={'depth': config['depth']})

        # When spikes are already int8 (the default stype), the per-timestep
        # int8 casts are redundant copies; skip them. For float/bfloat16 stype
        # the bitwise ops require an int8 operand, so the casts stay.
        self._spike_is_int8 = (self.stype == torch.int8)


    def reset(self, verbose=False):
        self.timestep_cur = 0
        super().reset(verbose)


    def forward(self, input: torch.tensor):
        self.tick()
        # input is a spike tensor
        input_d = self.dff(input)
        if self._spike_is_int8:
            # operands already int8: skip the redundant casts and the result cast.
            a, b = input, input_d
        else:
            a, b = input.type(torch.int8), input_d.type(torch.int8)
        if self.polarity == 'unipolar':
            out = a & b
        else:
            # XNOR as 1-(a^b): identical 0/1 result to (1-a)^b for spike
            # operands, and ~1.7x faster on MPS.
            out = 1 - (a ^ b)
        return out if self._spike_is_int8 else out.type(self.stype)


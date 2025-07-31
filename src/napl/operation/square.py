import torch

from napl.utils import *
from napl.base import napl_base
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
                'mode': 'bipolar',
                'depth': 1,
            }
        ):
        super().__init__(config, ['mode'], mode_required=True)

        # the depth of input
        self.dff = dff(config={'depth': config['depth']})
    

    def reset(self, verbose=False):
        self.timestep_cur = 0
        super().reset(verbose)
    

    def forward(self, input: torch.tensor):
        self.tick()
        # input is a spike tensor
        input_d = self.dff(input)
        if self.mode == 'unipolar':
            return (input.type(torch.int8) & input_d.type(torch.int8)).type(self.stype)
        else:
            return (1 - input.type(torch.int8) ^ input_d.type(torch.int8)).type(self.stype)


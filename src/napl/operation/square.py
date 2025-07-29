import torch

from napl.utils import *
from napl.base import napl_base
from napl.operation import delay


class square_delay(napl_base):
    """
    This module is for unary square with AND gate and delay, supporting unipolar/bipolar.
    References:
    1) uGEMM: Unary Computing Architecture for GEMM Applications
    2) uGEMM: Unary Computing for GEMM Applications
    """
    def __init__(
            self,
            config={
                'mode': 'bipolar',
                'delay': 1,
            }
        ):
        super().__init__()

        # check config
        check_config(config, ['mode'])
        self.mode = check_mode(config)
        self.name = check_name(config)

        # the delay of input
        self.delay = delay(config={'delay': config['delay']})
        
    
    def reset(self):
        self.delay.reset()

    
    def forward(self, input: torch.tensor):
        # input is a spike tensor
        input_d = self.delay(input)
        # print(input, input_d)
        if self.mode == 'unipolar':
            return (input.type(torch.int8) & input_d.type(torch.int8)).type(self.stype)
        else:
            return (1 - input.type(torch.int8) ^ input_d.type(torch.int8)).type(self.stype)


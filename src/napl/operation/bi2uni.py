import torch

from napl.utils import *
from napl.base import napl_base


class bi2uni(napl_base):
    """
    Convert bipolar spike trains to unipolar with non-scaled addition, please refer to
    'In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing'
    """
    def __init__(
            self, 
            config={
                'bitwidth' : 3,
            }
    ):
        super().__init__(config, ['bitwidth'], mode_required=False)

        # bitwidth of the accumulator
        self.bitwidth = config['bitwidth']
        # max value in the accumulator
        self.acc_max = 2**(self.bitwidth-1) - 1
        # min value in the accumulator
        self.acc_min = -2**(self.bitwidth-1)
        self.accumulator = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)


    def reset(self):
        """
        Reset the accumulator only.
        """
        self.accumulator.data = torch.zeros(1, dtype=self.ntype, device=self.accumulator.device)


    def forward(self, input):
        # calculate (2*input-1)/1
        # input bitstreams are [input, input, 0]
        # self.accumulator.data = self.accumulator.add(input.mul(2).sub(1).type(self.ntype)).clamp(self.acc_min, self.acc_max)
        self.accumulator.data = self.accumulator.add(input.type(self.ntype)*2 - 1).clamp(self.acc_min, self.acc_max)
        output = torch.ge(self.accumulator, 1).type(self.ntype)
        self.accumulator.sub_(output).clamp_(self.acc_min, self.acc_max)
        return output.type(self.stype)
    
    
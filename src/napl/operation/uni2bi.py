import torch

from napl.utils import *
from napl.base import napl_base


class uni2bi(napl_base):
    """
    Convert unipolar spike trains to bipolar with scaled addition, please refer to
    "In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing"
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
        # calculate (input+1)/2
        # input bitstreams are [input, 1]
        self.accumulator.data = self.accumulator.add(input.add(1).type(self.ntype)).clamp(self.acc_min, self.acc_max)
        output = torch.ge(self.accumulator, 2).type(self.ntype)
        self.accumulator.sub_(output.mul(2)).clamp_(self.acc_min, self.acc_max)
        return output.type(self.stype)
    

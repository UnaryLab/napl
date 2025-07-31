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
                'width' : 3,
            }
    ):
        super().__init__(config, ['width'], mode_required=False)

        # width of the accumulator
        self.width = config['width']
        # max value in the accumulator
        self.acc_max = 2**(self.width-1) - 1
        # min value in the accumulator
        self.acc_min = -2**(self.width-1)
        self.accumulator = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)


    def reset(self, verbose=False):
        """
        Reset the accumulator only.
        """
        self.timestep_cur = 0
        self.accumulator.data = torch.zeros(1, dtype=self.ntype, device=self.accumulator.device)


    def forward(self, input):
        self.tick()
        # calculate (input+1)/2
        # input spike trains are [input, 1]
        self.accumulator.data = self.accumulator.add(input.add(1).type(self.ntype)).clamp(self.acc_min, self.acc_max)
        output = torch.ge(self.accumulator, 2).type(self.ntype)
        self.accumulator.sub_(output.mul(2)).clamp_(self.acc_min, self.acc_max)
        return output.type(self.stype)
    

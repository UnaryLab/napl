import torch

from napl.utils import *
from napl.base import napl_base
from napl.operation import add_any


class relu_cnt(napl_base):
    """
    ReLU activation by comparing the spike value in a counter with bipolar 0 
    The spike train should always bipolar and rate coded
    """
    def __init__(
            self, 
            config={
                'width' : 3,
            }
    ):
        super().__init__(config, ['width'], polarity_required=False)

        self.width = config['width']

        self.buf_max = 2**self.width - 1
        self.buf_half = 2**(self.width - 1)
        self.acc = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype).fill_(2**(self.width - 1)), requires_grad=False)
    

    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.acc.data = torch.zeros(1, dtype=self.ntype).fill_(2**(self.width - 1))


    def forward(self, input):
        self.tick()
        # check whether acc is larger than or equal to half.
        half_prob_flag = torch.ge(self.acc, self.buf_half)
        # only when input is 0 and flag is 1, output 0; otherwise 1
        output = input.type(torch.int8) | (1 - half_prob_flag.type(torch.int8))
        # update the accumulator based on output, thus acc update is after output generation
        self.acc.data = self.acc.add(output.mul(2).sub(1).type(self.ntype)).clamp(0, self.buf_max)
        return output.type(self.stype)


class relu_sat(napl_base):
    """
    ReLU activation by saturating the spike value to 0
    The spike train should always bipolar and rate coded
    """
    def __init__(
            self, 
            config={
                'width' : 3,
            }
    ):
        super().__init__(config, ['width'], polarity_required=False)

        self.width = config['width']

        self.sub_1 = add_any({'polarity': 'bipolar', 'scale': 1, 'width': self.width})
        self.add_1 = add_any({'polarity': 'bipolar', 'scale': 1, 'width': self.width})
    

    def reset(self, verbose=False):
        self.timestep_cur = 0
        super().reset(verbose)


    def forward(self, input):
        self.tick()
        # check whether acc is larger than or equal to half.
        sub_1_out = self.sub_1(torch.stack([input, torch.zeros_like(input)], dim=0), dim=0)
        output = self.add_1(torch.stack([sub_1_out, torch.ones_like(input)], dim=0), dim=0)
        return output
    

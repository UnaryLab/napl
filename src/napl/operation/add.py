import torch

from napl.utils import *
from napl.base import napl_base


class add_any(napl_base):
    """
    This module is any-scale addition.
    Supported polarity: unipolar/bipolar.
    """
    def __init__(
            self, 
            config={
                'polarity' : 'bipolar', 
                'scale' : 2,
                'width' : 10,
            }
        ):
        super().__init__(config, ['polarity', 'scale', 'width'], mode_required=True)

        # width of the accumulator
        self.width = config['width']
        # max value in the accumulator
        self.acc_max = 2**(self.width-1) - 1
        # min value in the accumulator
        self.acc_min = -2**(self.width-1)
        
        # the carry scale at the output
        self.scale = torch.nn.Parameter(torch.tensor(config['scale'], dtype=self.ntype), requires_grad=False)
        # accumulation offset
        self.offset = 0
        # accumulator for (PC - offset)
        self.accumulator = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)
        self.is_first_call = True


    def reset(self, verbose=False):
        """
        Reset the accumulator only.
        """
        self.timestep_cur = 0
        self.accumulator.data = torch.zeros(1, dtype=self.ntype, device=self.accumulator.device)
        self.is_first_call = True


    def forward(self, input, entry=None, dim=-1):
        self.tick()
        if self.is_first_call:
            if entry is None:
                entry = input.size()[dim]

            if self.polarity == 'bipolar':
                self.offset = (entry - self.scale)/2

            self.is_first_call = False

        acc_delta = torch.sum(input.type(self.ntype), dim) - self.offset
        self.accumulator.data = self.accumulator.add(acc_delta).clamp(self.acc_min, self.acc_max)
        output = torch.ge(self.accumulator, self.scale).type(self.ntype)
        self.accumulator.sub_(output * self.scale).clamp_(self.acc_min, self.acc_max)
        return output.type(self.stype)


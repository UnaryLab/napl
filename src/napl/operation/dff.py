import torch

from napl.utils import *
from napl.base import napl_base


class dff(napl_base):
    """
    This module is for d flip flop
    """
    def __init__(
            self,
            config={'depth': 1}
        ):
        super().__init__(config, ['depth'], mode_required=False)

        self.depth = config['depth']
        self.reg = torch.nn.Parameter(torch.zeros(self.depth, dtype=self.stype), requires_grad=False)
        self.is_first_call = True

    
    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.reg.data = torch.zeros(self.depth, dtype=self.stype, device=self.reg.device)
        self.is_first_call = True

    
    def forward(self, input: torch.tensor):
        self.tick()
        # input is a spike tensor
        if self.is_first_call:
            input_shape = list(input.shape)
            input_shape.insert(0, self.depth)
            self.reg.data = torch.zeros(input_shape, dtype=self.stype, device=self.reg.device)
            self.is_first_call = False

        output = self.reg.data[0]
        self.reg.data = torch.roll(self.reg, -1, dims=0)
        self.reg.data[-1] = input.clone().detach()
        return output


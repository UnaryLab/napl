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
        self.input_d = torch.nn.Parameter(torch.zeros(self.depth, dtype=self.stype), requires_grad=False)
        self.is_first_call = True

    
    def reset(self, verbose=False):
        self.input_d.data = torch.zeros(self.depth, dtype=self.stype, device=self.input_d.device)
        self.is_first_call = True

    
    def forward(self, input: torch.tensor):
        # input is a spike tensor
        if self.is_first_call:
            input_shape = list(input.shape)
            input_shape.insert(0, self.depth)
            self.input_d.data = torch.zeros(input_shape, dtype=self.stype, device=self.input_d.device)
            self.is_first_call = False

        output = self.input_d.data[-1]
        self.input_d.data = torch.roll(self.input_d, 1, dims=0)
        self.input_d.data[0] = input.clone().detach()
        return output


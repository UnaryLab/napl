import torch

from napl.utils import *
from napl.base import napl_base


class delay(napl_base):
    """
    This module is for 1 cycle delay with a register
    """
    def __init__(
            self,
            config={'delay': 1}
        ):
        super().__init__(config, ['delay'], mode_required=False)

        self.delay = config['delay']
        self.input_d = torch.nn.Parameter(torch.zeros(self.delay, dtype=self.stype), requires_grad=False)
        self.is_first_call = True

    
    def reset(self):
        self.input_d.data = torch.zeros(self.delay, dtype=self.stype, device=self.input_d.device)
        self.is_first_call = True

    
    def forward(self, input: torch.tensor):
        # input is a spike tensor
        if self.is_first_call:
            input_shape = list(input.shape)
            input_shape.insert(0, self.delay)
            self.input_d.data = torch.zeros(input_shape, dtype=self.stype, device=self.input_d.device)
            self.is_first_call = False

        output = self.input_d.data[-1]
        self.input_d.data = torch.roll(self.input_d, 1, dims=0)
        self.input_d.data[0] = input.clone().detach()
        return output


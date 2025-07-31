import torch

from napl.utils import *
from napl.base import napl_base
from napl.operation import add_any


class sigmoid_hard(napl_base):
    """
    This is a scaled addition (input+1)/2.
    It works for both unipolar and bipolar spike trains.
    """
    def __init__(
        self, 
        config={
            'polarity' : 'bipolar'
        }, 
    ):
        super().__init__(config, ['polarity'], polarity_required=True)

        self.scaled_add = add_any({
            'polarity': self.polarity, 
            'scale' : 2,
            'width' : 3,
            })


    def reset(self, verbose=False):
        self.timestep_cur = 0
        super().reset(verbose)
    

    def forward(self, input: torch.tensor):
        self.tick()
        return self.scaled_add(torch.stack([input, torch.ones_like(input)], dim=0), dim=0)


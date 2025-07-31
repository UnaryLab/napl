import torch

from napl.utils import *
from napl.base import napl_base


class tanh_hard(napl_base):
    """
    This is a fsu scaled addition (input+1)/2.
    It works for both unipolar and bipolar spike trains.
    """
    def __init__(
        self, 
        config={}, 
    ):
        super().__init__(config, [], polarity_required=False)


    def reset(self, verbose=False):
        self.timestep_cur = 0
    

    def forward(self, input: torch.tensor):
        self.tick()
        return input


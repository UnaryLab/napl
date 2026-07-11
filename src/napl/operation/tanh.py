import torch

from napl.utils import *
from napl.base import napl_base, hw_params


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
        self.hw = hw_params(pp_delay=0)


    def reset(self, verbose=False):
        self.timestep_cur = 0


    def forward(self, input: torch.tensor):
        self.tick()
        return input


class tanh_hub(napl_base):
    """
    Binary-domain hard tanh: clip the input to [-1, 1] (a Hardtanh), the binary-domain
    counterpart used for training/inference. Single-shot, no tick.
    """
    def __init__(
        self,
        config={},
    ):
        super().__init__(config, [])
        self.delay = 0

    def forward(self, input: torch.tensor):
        return torch.nn.functional.hardtanh(input, -1.0, 1.0)


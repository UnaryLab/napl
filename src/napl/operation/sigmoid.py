import torch

from napl.utils import *
from napl.base import napl_base, hw_params
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
        self.hw = hw_params(pp_delay=0)

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
        # (input+1)/2: feed the pre-reduced per-timestep sum (input+1) directly
        # (no all-ones stack); entry=2 is the reduced operand count [input, 1]
        # used for the bipolar offset.
        return self.scaled_add(input + 1, dim=None, entry=2)


class sigmoid_hub(napl_base):
    """
    Binary-domain hard sigmoid: Hardsigmoid(input * scale), a piecewise-linear
    approximation of the sigmoid. Single-shot, no tick. Default scale 3.
    """
    def __init__(
        self,
        config={
            'scale': 3,
        },
    ):
        super().__init__(config, [])
        self.delay = 0
        self.scale = config.get('scale', 3)

    def forward(self, input: torch.tensor):
        return torch.nn.functional.hardsigmoid(input * self.scale)


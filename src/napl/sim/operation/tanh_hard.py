import torch

from napl.sim.base import napl_base, hw_params


class tanh_hard(napl_base):
    """
    This is a streaming scaled addition (input+1)/2.
    It works for both unipolar and bipolar spike streams.
    """
    def __init__(
        self,
        config={},
    ):
        super().__init__(config, [], polarity_required=False)
        self.hw = hw_params(pp_delay=0)


    def forward(self, input: torch.tensor):
        return input

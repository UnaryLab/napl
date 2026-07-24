import torch

from napl.sim.base import napl_base, hw_params


class max_tc(napl_base):
    """
    This class returns the max using OR gate.
    Temporal-coded signals always start with 1s, followed by 0s.
    """
    def __init__(
            self,
            config = {}
    ):
        super().__init__(config, [], polarity_required=False)
        self.hw = hw_params(pp_delay=0)


    def forward(self, input_0, input_1):
        output = input_0.type(torch.int8) | input_1.type(torch.int8)
        return output.type(self.stype)

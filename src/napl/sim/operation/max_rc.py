import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import sync_skewed


class max_rc(napl_base):
    """
    This class returns the max and argmax using sync_skewed.
    """
    def __init__(
            self,
            config = {}
    ):
        super().__init__(config, [], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        self.dff = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        # default to optimal width
        self.sync = sync_skewed({'width': 2})


    def _reset(self):
        self.dff.data = torch.zeros(1, dtype=torch.int8, device=self.dff.device)


    def forward(self, input_0, input_1):
        # sync input_0 to input_1
        sync_0, sync_1 = self.sync(input_0, input_1)
        sync_0_i8 = sync_0.type(torch.int8)
        sync_1_i8 = sync_1.type(torch.int8)
        # if sync_0/1 is 01 or 10, enable dff update
        d_enable = sync_0_i8 ^ sync_1_i8

        # generate output before the dff update
        # if self.dff == 1, input_1 is larger, and max is 1
        # mux(dff, input_1, input_0) with fewer elementwise ops (dff is {0,1})
        output = input_0 + self.dff * (input_1 - input_0)

        # update the dff if d_enable is 1: mux(d_enable, sync_1, dff)
        # sync_0/1 is 01, meaning input_0 < input_1
        # this dff value also indicates argmax
        self.dff.data = self.dff + d_enable * (sync_1_i8 - self.dff)

        # if self.dff == 1, input_1 is larger, and max is 1
        return output.type(self.stype), self.dff.type(self.stype)

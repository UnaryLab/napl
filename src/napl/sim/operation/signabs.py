import torch

from napl.utils import *
from napl.sim.base import napl_base, hw_params


class signabs(napl_base):
    """
    This module
    1) calculates the sign and magnitude of bipolar spikes.
    2) works for rate coding only.
    An output sign bit of 0 means positive at the current timestep.
    An output sign bit of 1 means negative at the current timestep.
    """
    def __init__(
            self,
            config={
                'width' : 3,
            }
    ):
        super().__init__(config, ['width'], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        self.width = config['width']

        self.acc_max = 2**self.width - 1
        self.acc_med = 2**(self.width - 1)
        self.acc = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype).fill_(self.acc_med), requires_grad=False)


    def _reset(self):
        self.acc.data = torch.zeros(1, dtype=self.ntype, device=self.acc.device).fill_(self.acc_med)


    def forward(self, input):
        # update the accumulator based on input: +1 for input 1; -1 for input 0
        # the accumulator saturates at min and max
        # acc + 2*input - 1 fused via alpha; in-place only after the first-call (1,)->(N,) broadcast
        if self.acc.shape == input.shape:
            self.acc.data.add_(input, alpha=2).sub_(1).clamp_(0, self.acc_max)
        else:
            self.acc.data = self.acc.add(input, alpha=2).sub_(1).clamp_(0, self.acc_max)
        sign = torch.lt(self.acc, self.acc_med).type(torch.int8)
        abs = sign ^ input.type(torch.int8)
        return sign.type(self.stype), abs.type(self.stype)

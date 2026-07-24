import torch

from napl.utils import *
from napl.sim.base import napl_base, hw_params


class bi2uni(napl_base):
    """
    Convert bipolar spike streams to unipolar with non-scaled addition, please refer to
    'In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing'
    """
    def __init__(
            self,
            config={
                'width' : 2,
            }
    ):
        super().__init__(config, ['width'], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        # width of the accumulator
        self.width = config['width']
        # max value in the accumulator
        self.acc_max = 2**(self.width-1) - 1
        # min value in the accumulator
        self.acc_min = -2**(self.width-1)
        self.accumulator = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)


    def _reset(self):
        """
        Reset the accumulator only.
        """
        self.accumulator.data = torch.zeros(1, dtype=self.ntype, device=self.accumulator.device)


    def forward(self, input):
        # calculate (2*input-1)/1
        # input spike streams are [input, input, 0]
        # fuse acc + (2*input - 1), then clamp. ntype destination promotes the int8 input
        # (alpha=2), so the explicit .type(ntype) cast is unnecessary.
        acc = self.accumulator
        if acc.shape == input.shape:
            # steady state: update the accumulator buffer in place (no per-timestep alloc).
            acc = acc.add_(input, alpha=2).sub_(1).clamp_(self.acc_min, self.acc_max)
        else:
            # first call: broadcast the (1,) accumulator up to the input shape out of place.
            acc = acc.add(input, alpha=2).sub_(1).clamp_(self.acc_min, self.acc_max)
        # output as stype directly; acc.sub_ promotes int8 up to float32 destination (no truncation),
        # which drops the separate stype cast at return.
        output = torch.ge(acc, 1).type(self.stype)
        # acc is already in [acc_min, acc_max] and output in {0,1} with output==1 only when acc>=1,
        # so acc-output stays in range; the trailing clamp is a provable no-op.
        acc.sub_(output)
        self.accumulator.data = acc
        return output

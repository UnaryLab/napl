import torch

from napl.sim.base import napl_base, hw_params


class relu_cnt(napl_base):
    """
    ReLU activation by comparing the spike value in a counter with bipolar 0
    The spike stream should always bipolar and rate coded
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

        self.buf_max = 2**self.width - 1
        self.buf_half = 2**(self.width - 1)
        self.acc = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype).fill_(2**(self.width - 1)), requires_grad=False)


    def _reset(self):
        self.acc.data = torch.zeros(1, dtype=self.ntype, device=self.acc.device).fill_(2**(self.width - 1))


    def forward(self, input):
        # below_half is the complement of (acc >= half); lt avoids the extra (1 - ge) step.
        below_half = torch.lt(self.acc, self.buf_half)
        # only when input is 0 and flag is 1, output 0; otherwise 1
        # int8 | bool yields int8, so below_half needs no separate cast
        output = input.type(torch.int8) | below_half
        # update the accumulator based on output, thus acc update is after output generation
        # acc += 2*output - 1, then clamp; fused/in-place to avoid per-timestep intermediate allocations
        self.acc.data = self.acc.add(output, alpha=2).sub_(1).clamp_(0, self.buf_max)
        return output.type(self.stype)

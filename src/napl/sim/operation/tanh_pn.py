import torch

from napl.sim.base import napl_base, hw_params


class tanh_pn(napl_base):
    """
    FSM-based tanh(N*x/2) with N = 2**depth states, from
    "Stochastic neural computation I: Computational elements".
    The spike stream should always be bipolar and rate coded.
    """
    def __init__(
            self,
            config={
                'depth' : 5,
            }
    ):
        super().__init__(config, ['depth'], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        self.depth = config['depth']

        self.cnt_max = 2**self.depth - 1
        self.cnt_half = 2**(self.depth - 1)
        # scalar state; broadcasts up to the input shape on the first forward()
        self.cnt = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype).fill_(self.cnt_half), requires_grad=False)


    def _reset(self):
        self.cnt.data = torch.zeros(1, dtype=self.ntype, device=self.cnt.device).fill_(self.cnt_half)


    def forward(self, input):
        # output looks at the state before this timestep's update
        output = torch.ge(self.cnt, self.cnt_half).type(self.stype).expand_as(input)
        # cnt += 2*input - 1, then clamp; in-place once shape matches (first call
        # broadcasts (1,) -> input shape out-of-place)
        if self.cnt.shape == input.shape:
            self.cnt.data.add_(input, alpha=2).sub_(1).clamp_(0, self.cnt_max)
        else:
            self.cnt.data = self.cnt.add(input.type(self.ntype), alpha=2).sub_(1).clamp_(0, self.cnt_max)
        return output

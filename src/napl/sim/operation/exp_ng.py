import torch

from napl.sim.base import napl_base, hw_params


class exp_ng(napl_base):
    """
    FSM-based exponentiation exp(-2*gain*x) with a saturating up/down counter, from
    "Stochastic neural computation I: Computational elements" (Brown and Card).
    The input spike stream should always be bipolar and rate coded, with a non-negative
    value; the output spike stream is unipolar.
    """
    def __init__(
            self,
            config={
                'depth': 5,
                'gain': 1,
            }
    ):
        super().__init__(config, ['depth'], polarity_required=False)
        # output is combinational from the state counter; input reaches it one cycle later
        self.hw = hw_params(pp_delay=1)

        self.depth = config['depth']
        self.gain = config.get('gain', 1)

        self.cnt_max = 2**self.depth - 1
        # emit a 1-spike while the counter is below this threshold (top `gain` states emit 0)
        self.thd = 2**self.depth - self.gain
        self.cnt = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype).fill_(2**(self.depth - 1)), requires_grad=False)


    def _reset(self):
        self.cnt.data = torch.zeros(1, dtype=self.ntype, device=self.cnt.device).fill_(2**(self.depth - 1))


    def forward(self, input):
        # output reflects the state before this timestep's input is absorbed
        output = torch.lt(self.cnt, self.thd).type(self.stype)
        if output.shape != input.shape:
            output = torch.zeros_like(input) + output
        # count up on a 1-spike, down on a 0-spike, saturating at [0, 2**depth - 1]
        if self.cnt.shape == input.shape:
            # steady state: in-place on the ntype counter (promotion is a no-op)
            self.cnt.data.add_(input, alpha=2).sub_(1).clamp_(0, self.cnt_max)
        else:
            # first call: out-of-place add broadcasts the scalar counter to input shape
            self.cnt.data = self.cnt.add(input.type(self.ntype), alpha=2).sub_(1).clamp_(0, self.cnt_max)
        return output

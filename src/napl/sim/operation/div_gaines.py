import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.module import gen_num_seq


class div_gaines(napl_base):
    """
    Gaines division: a saturating up/down counter integrates the division error and
    drives the quotient stream by comparison against an RNG sequence.
    Unipolar computes dividend/divisor; bipolar computes the same on [-1, 1].
    Reference:
    1) B. R. Gaines, 'Stochastic Computing Systems', 1969
    """
    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
            # depth of the saturating counter; experiments in UnarySim default to 5
            'depth' : 5,
            'generator' : 'Sobol',
            'dim' : 1,
        }
    ):
        super().__init__(config, ['polarity', 'depth', 'generator'], polarity_required=True)
        # quotient spike is a comparison against the counter register, so one-cycle latency
        self.hw = hw_params(pp_delay=1)

        self.depth = config['depth']
        config['width'] = self.depth

        # rng sequence scaled to integers in [0, 2**depth), matching the counter range
        # static python list of thresholds; indexing with a python scalar per timestep
        # avoids a device-scalar fetch (a per-timestep GPU sync)
        self.rng_seq = torch.floor(gen_num_seq(config).mul(2 ** self.depth)).tolist()
        # index of numbers in the rng seq
        self.idx = 0

        self.scnt_max = 2 ** self.depth - 1
        self.scnt_init = 2 ** (self.depth - 1)
        # saturating up/down counter; scalar that broadcasts to the input shape on the first forward
        self.scnt = torch.nn.Parameter(torch.full((1,), float(self.scnt_init), dtype=self.ntype), requires_grad=False)
        # previous divisor spike, used to decorrelate the counter feedback in bipolar mode
        self.divisor_d = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)


    def _reset(self):
        self.idx = 0
        self.scnt.data = torch.full((1,), float(self.scnt_init), dtype=self.ntype, device=self.scnt.device)
        self.divisor_d.data = torch.zeros(1, dtype=torch.int8, device=self.divisor_d.device)


    def forward(self, dividend, divisor):

        # quotient spike from the counter state; identical for both polarities
        output = torch.gt(self.scnt, self.rng_seq[self.idx]).type(torch.int8)
        self.idx = (self.idx + 1) % len(self.rng_seq)
        if output.shape != dividend.shape:
            # first timestep only: counter is still a scalar, expand is a free view
            output = output.expand(dividend.shape)

        if self.polarity == 'unipolar':
            # counter integrates dividend - output * divisor; 0/1 spikes are exact
            # under the int8 -> float promotion of the out-of-place add below
            inc = dividend
            dec = output & divisor.type(torch.int8)
        else:
            dividend_i8 = dividend.type(torch.int8)
            divisor_i8 = divisor.type(torch.int8)
            # XNOR terms: dividend*divisor and output*divisor (via the delayed divisor
            # spike, which decorrelates the feedback from the fresh divisor), folded
            # algebraically into inc - dec = XNOR(divisor_d^divisor^output) - (dividend^divisor)
            inc = (self.divisor_d ^ divisor_i8 ^ output) ^ 1
            dec = dividend_i8 ^ divisor_i8
            self.divisor_d.data = divisor_i8

        # saturating up/down counter; the out-of-place add broadcasts up to the input
        # shape on the first call, then sub_/clamp_ mutate that fresh tensor in place
        self.scnt.data = self.scnt.add(inc).sub_(dec).clamp_(0, self.scnt_max)

        return output.type(self.stype)

import torch, math

from napl.sim.base import napl_base, hw_params
from napl.sim.module import gen_num_seq
from loguru import logger


class div_cordiv(napl_base):
    """
    The divivison using correlated divivison, for unipolar only
    The dividend and divisor have to be synchronized before fed to this kernel
    Reference:
    1) 'Design of Division Circuits for Stochastic Computing'
    2) 'In-Stream Stochastic Division and Square Root via Correlation'
    3) 'In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing'
    """
    def __init__(
        self,
        config={
            # experiments shows that depth of 2 is the best for accuracy
            'depth' : 2,
            'generator' : 'Sobol',
        }
    ):
        super().__init__(config, ['depth', 'generator'], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        self.depth = config['depth']
        assert math.log2(self.depth) == math.ceil(math.log2(self.depth)), logger.error(f'Input depth <{self.depth}> is not power of 2.')
        self.width = int(math.log2(self.depth))
        config['width'] = self.width

        # rand sequence to choose q
        self.rand_seq = torch.nn.Parameter(torch.floor(gen_num_seq(config).mul(self.depth)).type(torch.long), requires_grad=False)
        # static list of buffer-row indices; lets forward() index buffer_q with a
        # Python int (a cheap view) instead of a device scalar tensor (a per-timestep GPU sync)
        self.rand_seq_idx = self.rand_seq.tolist()
        # index of numbers in the rand seq
        self.idx = 0

        # the buffer to save a few q
        self.buffer_q = torch.nn.Parameter(torch.zeros(self.depth, dtype=self.stype), requires_grad=False)

        self.is_first_call = True


    def _reset(self):
        self.idx = 0
        self.buffer_q.data = torch.zeros(self.depth, dtype=self.stype, device=self.buffer_q.device)
        self.is_first_call = True


    def forward(self, dividend, divisor):
        if self.is_first_call:
            dividend_shape = list(dividend.shape)
            divisor_shape = list(divisor.shape)
            if len(dividend_shape) > len(divisor_shape):
                input_shape = dividend_shape
            else:
                input_shape = divisor_shape
            input_shape.insert(0, self.depth)
            self.buffer_q.data = torch.zeros(input_shape, dtype=self.stype, device=self.buffer_q.device)
            self.is_first_call = False

        # generate the random number to index buffer_q
        # always generating, no need to deal with conditional probability
        # rand_q is a per-call local (read only on the next line), so keep it off self:
        # nn.Module.__setattr__ runs Parameter/Module/Tensor isinstance checks on every
        # assignment and is a measurable per-timestep cost here.
        divisor_eq_1 = torch.eq(divisor, 1)
        rand_q = self.buffer_q[self.rand_seq_idx[self.idx]]
        self.idx = (self.idx + 1) % self.depth

        # select dividend where the divisor spikes, else the buffered q (0/1 blend == where).
        # where() of two stype operands is already stype, so no .type(self.stype) cast needed.
        quotient = torch.where(divisor_eq_1, dividend, rand_q).view(dividend.size())

        # buffer_q update: shift in the new quotient only where divisor is a valid spike.
        # equivalent to roll(+1) with row 0 := quotient, then where(divisor_eq_1, shifted, old);
        # done as in-place per-row shifts (top-down so each row reads its un-updated source),
        # avoiding the two full [depth, *shape] allocations of the roll + where form.
        # out=buf[r] fuses the where + slice copy_ into one kernel; out fully aliases
        # input buf[r] (legal in-place), and rows are disjoint slices (no partial overlap).
        buf = self.buffer_q.data
        for r in range(self.depth - 1, 0, -1):
            torch.where(divisor_eq_1, buf[r - 1], buf[r], out=buf[r])
        torch.where(divisor_eq_1, quotient, buf[0], out=buf[0])

        return quotient

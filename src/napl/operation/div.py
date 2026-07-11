import torch, math

from napl.utils import *
from napl.base import napl_base, hw_params
from napl.module import gen_num_seq
from napl.operation import uni2bi, bi2uni, sign_abs, sync_skewed
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


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.idx = 0
        self.buffer_q.data = torch.zeros(self.depth, dtype=self.stype, device=self.buffer_q.device)
        self.is_first_call = True


    def forward(self, dividend, divisor):
        self.tick()
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
        buf = self.buffer_q.data
        for r in range(self.depth - 1, 0, -1):
            buf[r] = torch.where(divisor_eq_1, buf[r - 1], buf[r])
        buf[0] = torch.where(divisor_eq_1, quotient, buf[0])

        return quotient


class div_iscb(napl_base):
    """
    This module is for in-stream correlation based division (iscbdiv) using rate coding. Please refer to
    1) 'In-Stream Stochastic Division and Square Root via Correlation'
    2) 'In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing'
    """
    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
        }
    ):
        super().__init__(config, ['polarity'], polarity_required=True)
        self.hw = hw_params(pp_delay=0)

        # fix width to optimal 3
        self.sync = sync_skewed({'width': 3})

        # for cordiv kernel, the config is fixed to optimal directly
        # this actually leads to 01 sequence
        self.cordiv_kernel = div_cordiv({'depth': 2, 'generator': 'sobol'})

        if self.polarity == 'bipolar':
            # fix width to optimal 3
            self.sign_abs_dividend = sign_abs({'width': 3})
            self.sign_abs_divisor  = sign_abs({'width': 3})
            # fix width to optimal 2
            self.bi2uni_dividend = bi2uni({'width': 2})
            self.bi2uni_divisor  = bi2uni({'width': 2})
            # fix width to optimal 3
            self.uni2bi_quotient = uni2bi({'width': 3})


    def reset(self, verbose=False):
        self.timestep_cur = 0
        super().reset(verbose)


    def bipolar_forward(self, dividend: torch.tensor, divisor: torch.tensor):
        # dividend and divisor are both spike tensors
        sign_dividend, abs_dividend = self.sign_abs_dividend(dividend)
        sign_divisor, abs_divisor = self.sign_abs_divisor(divisor)
        uni_abs_dividend = self.bi2uni_dividend(abs_dividend)
        uni_abs_divisor = self.bi2uni_divisor(abs_divisor)
        uni_abs_quotient = self.unipolar_forward(uni_abs_dividend, uni_abs_divisor)
        bi_abs_quotient = self.uni2bi_quotient(uni_abs_quotient)
        bi_quotient = sign_dividend.type(torch.int8) ^ sign_divisor.type(torch.int8) ^ bi_abs_quotient.type(torch.int8)
        return bi_quotient


    def unipolar_forward(self, dividend: torch.tensor, divisor: torch.tensor):
        # dividend and divisor are both spike tensors
        dividend_sync, divisor_sync = self.sync(dividend, divisor)
        quotient = self.cordiv_kernel(dividend_sync, divisor_sync)
        return quotient


    def forward(self, dividend, divisor):
        self.tick()
        if self.polarity == 'bipolar':
            output = self.bipolar_forward(dividend, divisor)
        else:
            output = self.unipolar_forward(dividend, divisor)
        return output.type(self.stype)


import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import uni2bi, bi2uni, signabs, sync_skewed, div_cordiv


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
            self.signabs_dividend = signabs({'width': 3})
            self.signabs_divisor  = signabs({'width': 3})
            # fix width to optimal 2
            self.bi2uni_dividend = bi2uni({'width': 2})
            self.bi2uni_divisor  = bi2uni({'width': 2})
            # fix width to optimal 3
            self.uni2bi_quotient = uni2bi({'width': 3})


    def bipolar_forward(self, dividend: torch.tensor, divisor: torch.tensor):
        # dividend and divisor are both spike tensors
        sign_dividend, abs_dividend = self.signabs_dividend(dividend)
        sign_divisor, abs_divisor = self.signabs_divisor(divisor)
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
        if self.polarity == 'bipolar':
            output = self.bipolar_forward(dividend, divisor)
        else:
            output = self.unipolar_forward(dividend, divisor)
        return output.type(self.stype)

import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import bi2uni, div_cordiv


class sqrt_traceiscb(napl_base):
    """
    This module is for square root via stochastic bit inserting using iscbdiv, supporting unipolar/bipolar.
    References:
    1) 'In-Stream Stochastic Division and Square Root via Correlation'
    2) 'In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing'
    """
    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
        },
    ):
        super().__init__(config, ['polarity'], polarity_required=True)
        self.hw = hw_params(pp_delay=0)

        # for cordiv kernel, the config is fixed to optimal directly
        # this actually leads to 01 sequence
        self.cordiv_kernel = div_cordiv({'depth': 2, 'generator': 'sobol'})
        self.dff = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        self.trace = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)

        if self.polarity == 'bipolar':
            # fix width to optimal 2
            self.bi2uni = bi2uni({'width': 2})


    def _reset(self):
        self.dff.data = torch.zeros(1, dtype=torch.int8, device=self.dff.device)
        self.trace.data = torch.zeros(1, dtype=torch.int8, device=self.dff.device)


    def unipolar_trace(self, output):
        dff_inv = 1 - self.dff
        dividend = dff_inv & output.type(torch.int8)
        divisor = self.dff | dividend

        # use actual quotient as trace
        self.trace.data = self.cordiv_kernel(dividend, divisor)

        self.dff.data = dff_inv


    def forward(self, input):
        trace = self.trace
        # for trace, input in {0,1}, ((1-trace) & input) + trace == trace | input:
        # one fused OR instead of sub/and/add temporaries per timestep
        output = (trace | input.type(torch.int8)).type(self.stype)
        if self.polarity == 'unipolar':
            # P_trace = P_out/(P_out+1)
            self.unipolar_trace(output)
        else:
            # P_trace = (P_out*2-1)/((P_out*2-1)+1)
            out = self.bi2uni(output)
            self.unipolar_trace(out)
        return output

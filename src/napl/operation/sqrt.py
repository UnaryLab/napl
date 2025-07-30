import torch

from napl.utils import *
from napl.base import napl_base
from napl.operation import bi2uni, jkff, div_cordiv, add_any


class sqrt_tracejkff(napl_base):
    """
    This module is for square root via stochastic bit inserting using jkff, supporting unipolar/bipolar.
    References:
    1) 'In-Stream Stochastic Division and Square Root via Correlation'
    2) 'In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing'
    """
    def __init__(
        self, 
        config={
            'mode' : 'bipolar',
        },
    ):
        super().__init__(config, ['mode'], mode_required=True)

        self.jkff = jkff()
        if self.mode == 'bipolar':
            # fix width to optimal 2
            self.bi2uni = bi2uni({'width': 2})
    

    def unipolar_trace(self, output):
        self.jkff(output, torch.ones_like(output))


    def forward(self, input):
        trace = self.jkff.q
        output = (((1 - trace) & input.type(torch.int8)) + trace).type(self.stype)
        if self.mode == 'unipolar':
            # P_trace = P_out/(P_out+1)
            self.unipolar_trace(output)
        else:
            # P_trace = (P_out*2-1)/((P_out*2-1)+1)
            out = self.bi2uni(output)
            self.unipolar_trace(out)
        return output
    


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
            'mode' : 'bipolar',
        },
    ):
        super().__init__(config, ['mode'], mode_required=True)

        # for cordiv kernel, the config is fixed to optimal directly
        # this actually leads to 01 sequence
        self.cordiv_kernel = div_cordiv({'depth': 2, 'generator': 'sobol'})
        self.dff = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        self.trace = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        
        if self.mode == 'bipolar':
            # fix width to optimal 2
            self.bi2uni = bi2uni({'width': 2})
    

    def reset(self, verbose=False):
        self.cordiv_kernel.reset(verbose)
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
        output = (((1 - trace) & input.type(torch.int8)) + trace).type(self.stype)
        if self.mode == 'unipolar':
            # P_trace = P_out/(P_out+1)
            self.unipolar_trace(output)
        else:
            # P_trace = (P_out*2-1)/((P_out*2-1)+1)
            out = self.bi2uni(output)
            self.unipolar_trace(out)
        return output
    

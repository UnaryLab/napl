import torch

from napl.utils import *
from napl.base import napl_base
from napl.operation import bi2uni, jkff, div_cordiv, add_any


class sqrt_tracejk(napl_base):
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
    

    def forward(self, input):
        output = (((1 - self.jkff.q) & input.type(torch.int8)) + self.jkff.q).type(self.stype)
        if self.mode == 'unipolar':
            # P_trace = P_out/(P_out+1)
            self.jkff(output, torch.ones_like(output))
        else:
            # P_trace = (P_out*2-1)/((P_out*2-1)+1)
            out = self.bi2uni(output)
            self.jkff(out, torch.ones_like(out))
        return output
    

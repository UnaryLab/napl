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
    The accuracy of sqrt_tracejkff is more sensitive to input spike train randomness than sqrt_traceiscb
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
        if self.mode == 'bipolar':
            self.bi2uni.reset(verbose)


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
    


# class sqrt_emit(napl_base):
#     """
#     This module is for square root via opportunistic bit inserting, supporting unipolar/bipolar.
#     References:
#     1) 'In-Stream Stochastic Division and Square Root via Correlation'
#     2) 'In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing'
#     """
#     def __init__(
#         self, 
#         config={
#             'mode' : 'bipolar',
#         },
#     ):
#         super().__init__(config, ['mode'], mode_required=True)

#         self.emit_out = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)

#         self.nsadd = add_any({'mode': 'unipolar', 'scale': 1, 'width': 3})
#         self.sr = ShiftReg(hwcfg_sr, swcfg_sr)
#         self.rng = RNG(hwcfg_rng, swcfg_rng)()
#         self.idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
        
#         if self.mode == 'bipolar':
#             # fix width to optimal 2
#             self.bi2uni = bi2uni({'width': 2})
    

#     def reset(self, verbose=False):
#         self.cordiv_kernel.reset(verbose)
#         self.dff.data = torch.zeros(1, dtype=torch.int8, device=self.dff.device)
#         self.trace.data = torch.zeros(1, dtype=torch.int8, device=self.dff.device)


#     def unipolar_emit(self, output):
#         output_inv = 1 - output
#         output_inv_scrambled, dontcare = self.sr(output_inv, index=self.idx.item()%self.entry_sr)
#         emit_out = output_inv_scrambled & output
#         return emit_out
    

#     def bipolar_emit(self, output):
#         output_inv = 1 - output
#         output_inv_scrambled, dontcare = self.sr(output_inv, index=self.idx.item()%self.entry_sr)
#         output_uni = self.bi2uni_emit(output)
#         emit_out = output_inv_scrambled & output_uni
#         return emit_out


#     def forward(self, input):
#         if list(self.emit_out.size()) != list(input.size()):
#                 self.emit_out.data = torch.zeros_like(input).type(torch.int8)
#             in_stack = torch.stack([input.type(torch.int8), self.emit_out], dim=0)
#             output = self.nsadd(in_stack)
#             if self.mode == "bipolar":
#                 self.emit_out.data = self.bipolar_emit(output)
#             else:
#                 self.emit_out.data = self.unipolar_emit(output)

#         trace = self.trace
#         output = (((1 - trace) & input.type(torch.int8)) + trace).type(self.stype)
#         if self.mode == 'unipolar':
#             # P_trace = P_out/(P_out+1)
#             self.unipolar_trace(output)
#         else:
#             # P_trace = (P_out*2-1)/((P_out*2-1)+1)
#             out = self.bi2uni(output)
#             self.unipolar_trace(out)
        
#         return output
    

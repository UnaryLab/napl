import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import bi2uni, add_any, shiftreg


class sqrt_emit(napl_base):
    """
    This module is for square root via opportunistic bit inserting, supporting unipolar/bipolar.
    References:
    1) 'In-Stream Stochastic Division and Square Root via Correlation'
    2) 'In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing'
    This module has best accuracy among all.
    """
    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
        },
    ):
        super().__init__(config, ['polarity'], polarity_required=True)
        self.hw = hw_params(pp_delay=0)

        self.emit_out = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)

        # a non-scaled add
        self.nsadd = add_any({'polarity': 'unipolar', 'scale': 1, 'width': 3})
        self.depth = 2
        self.shiftreg = shiftreg({'depth': self.depth})

        if self.polarity == 'bipolar':
            # fix width to optimal 2
            self.bi2uni = bi2uni({'width': 2})

        self.is_first_call = True


    def _reset(self):
        self.emit_out.data = torch.zeros(1, dtype=torch.int8, device=self.emit_out.device)
        self.is_first_call = True


    def unipolar_emit(self, output):
        output_inv = 1 - output
        output_inv_scrambled = self.shiftreg(output_inv)
        emit_out = output_inv_scrambled.type(torch.int8) & output.type(torch.int8)
        return emit_out


    def bipolar_emit(self, output):
        output_inv = 1 - output
        output_inv_scrambled = self.shiftreg(output_inv)
        output_uni = self.bi2uni(output)
        emit_out = output_inv_scrambled.type(torch.int8) & output_uni.type(torch.int8)
        return emit_out


    def forward(self, input):
        if self.is_first_call:
            self.emit_out.data = torch.zeros_like(input).type(self.stype)
            self.is_first_call = False

        # the 2-element reduction nsadd would do over a [2, N] stack is just the
        # elementwise int8 sum (both operands {0,1}, max 2, no overflow); compute it
        # directly and feed nsadd pre-reduced (dim=None) to drop the per-timestep
        # torch.stack allocation. nsadd re-casts to ntype, so the int8 partial is
        # bit-identical to torch.sum(stack, dtype=ntype).
        in_sum = input.type(torch.int8) + self.emit_out
        output = self.nsadd(in_sum, dim=None)
        if self.polarity == 'bipolar':
            self.emit_out.data = self.bipolar_emit(output)
        else:
            self.emit_out.data = self.unipolar_emit(output)

        return output

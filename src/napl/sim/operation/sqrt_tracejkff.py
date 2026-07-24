import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import bi2uni, jkff


class sqrt_tracejkff(napl_base):
    """
    This module is for square root via stochastic bit inserting using jkff, supporting unipolar/bipolar.
    References:
    1) 'In-Stream Stochastic Division and Square Root via Correlation'
    2) 'In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing'
    The accuracy of sqrt_tracejkff is more sensitive to input spike stream randomness than sqrt_traceiscb
    """
    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
        },
    ):
        super().__init__(config, ['polarity'], polarity_required=True)
        self.hw = hw_params(pp_delay=0)

        self.jkff = jkff()
        # constant K=1 input to the jkff, cached to avoid a per-timestep ones_like alloc
        self.jkff_k = None
        if self.polarity == 'bipolar':
            # fix width to optimal 2
            self.bi2uni = bi2uni({'width': 2})


    def _reset(self):
        self.jkff_k = None


    def unipolar_trace(self, output):
        k = self.jkff_k
        if k is None or k.shape != output.shape or k.device != output.device or k.dtype != output.dtype:
            k = torch.ones_like(output)
            self.jkff_k = k
        self.jkff(output, k)


    def forward(self, input):
        trace = self.jkff.q
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

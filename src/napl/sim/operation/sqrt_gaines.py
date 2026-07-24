import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.module import gen_num_seq


class sqrt_gaines(napl_base):
    """
    This module is for Gaines square root using a saturating up/down counter, supporting unipolar/bipolar.
    The output spike is the counter state compared against a random number sequence; the counter
    increments on input spikes and decrements when the output feedback indicates the squared output
    (output AND its 1-cycle delay for unipolar, output XNOR its delay for bipolar) spikes.
    Reference:
    1) 'Stochastic Computing Systems' (B. R. Gaines)
    """
    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
            # counter bit width; experiments in UnarySim use 5
            'width' : 5,
            'generator' : 'Sobol',
        },
    ):
        super().__init__(config, ['polarity', 'width', 'generator'], polarity_required=True)
        self.hw = hw_params(pp_delay=0)

        self.width = config['width']
        self.cnt_max = 2**self.width - 1
        self.cnt_half = 2**(self.width - 1)

        # random number sequence in [0, 2**width - 1] to threshold the counter state;
        # a static python list so forward() compares against a cheap python scalar
        # (a device scalar tensor would be a per-timestep GPU sync)
        self.rand_seq = torch.nn.Parameter(torch.floor(gen_num_seq(config).mul(2**self.width)), requires_grad=False)
        self.rand_seq_vals = self.rand_seq.tolist()
        self.idx = 0

        # saturating up/down counter, biased to half scale; broadcasts to input shape on first forward
        self.scnt = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype).fill_(self.cnt_half), requires_grad=False)
        # 1-cycle delayed output for the squared-output feedback
        self.out_d = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)


    def _reset(self):
        self.idx = 0
        self.scnt.data = torch.zeros(1, dtype=self.ntype, device=self.scnt.device).fill_(self.cnt_half)
        self.out_d.data = torch.zeros(1, dtype=torch.int8, device=self.out_d.device)


    def forward(self, input):
        # output spike from counter state vs the random threshold, same for both polarities
        output = torch.gt(self.scnt, self.rand_seq_vals[self.idx]).type(torch.int8)
        self.idx = (self.idx + 1) % len(self.rand_seq_vals)
        if output.shape != input.shape:
            # first call only: scnt is still scalar, so materialize output at the input shape
            output = output.expand(input.size()).contiguous()

        # counter increments on input spikes, decrements on the squared-output feedback
        if self.polarity == 'unipolar':
            # y*y in unipolar rate coding: output AND its 1-cycle delay
            dec = output & self.out_d
        else:
            # y*y in bipolar rate coding: output XNOR its 1-cycle delay
            dec = 1 - (output ^ self.out_d)
        self.out_d.data = output

        # spikes are 0/1, so the inc/dec muxes reduce to +input/-dec; one saturating clamp at the end
        self.scnt.data = (self.scnt + input - dec).clamp(0, self.cnt_max)

        return output.type(self.stype)

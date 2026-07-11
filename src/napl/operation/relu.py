import torch

from napl.utils import *
from napl.base import napl_base, hw_params
from napl.operation import add_any


class relu_cnt(napl_base):
    """
    ReLU activation by comparing the spike value in a counter with bipolar 0
    The spike train should always bipolar and rate coded
    """
    def __init__(
            self,
            config={
                'width' : 3,
            }
    ):
        super().__init__(config, ['width'], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        self.width = config['width']

        self.buf_max = 2**self.width - 1
        self.buf_half = 2**(self.width - 1)
        self.acc = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype).fill_(2**(self.width - 1)), requires_grad=False)


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.acc.data = torch.zeros(1, dtype=self.ntype, device=self.acc.device).fill_(2**(self.width - 1))


    def forward(self, input):
        self.tick()
        # below_half is the complement of (acc >= half); lt avoids the extra (1 - ge) step.
        below_half = torch.lt(self.acc, self.buf_half)
        # only when input is 0 and flag is 1, output 0; otherwise 1
        # int8 | bool yields int8, so below_half needs no separate cast
        output = input.type(torch.int8) | below_half
        # update the accumulator based on output, thus acc update is after output generation
        # acc += 2*output - 1, then clamp; fused/in-place to avoid per-timestep intermediate allocations
        self.acc.data = self.acc.add(output, alpha=2).sub_(1).clamp_(0, self.buf_max)
        return output.type(self.stype)


class relu_sat(napl_base):
    """
    ReLU activation by saturating the spike value to 0
    The spike train should always bipolar and rate coded
    """
    def __init__(
            self,
            config={}
    ):
        super().__init__(config, [], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        # default to optimal width
        self.sub_1 = add_any({'polarity': 'bipolar', 'scale': 1, 'width': 3})
        self.add_1 = add_any({'polarity': 'bipolar', 'scale': 1, 'width': 3})


    def reset(self, verbose=False):
        self.timestep_cur = 0
        super().reset(verbose)


    def forward(self, input):
        self.tick()
        # sub_1 moves input from [-1, 1] to [-1, 0]: the 2-row reduction sum([input, 0])
        # is just `input`, so feed the pre-reduced partial sum (dim=None) with the row
        # count as <entry> and skip the per-timestep stack alloc + sum reduction.
        sub_1_out = self.sub_1(input, entry=2, dim=None)
        # add_1 moves input from [-1, 0] to [0, 1]: sum([sub_1_out, 1]) == sub_1_out + 1
        output = self.add_1(sub_1_out + 1, entry=2, dim=None)
        return output


class relu_hub(napl_base):
    """
    Binary-domain ReLU: clip the input to [0, scale] (a Hardtanh). Single-shot, no tick;
    trainable for binary-domain quant-aware training.
    """
    def __init__(
            self,
            config={
                'scale': 1.0,
            }
        ):
        super().__init__(config, [])
        self.scale = config.get('scale', 1.0)

    def forward(self, input):
        return torch.nn.functional.hardtanh(input, 0.0, self.scale)


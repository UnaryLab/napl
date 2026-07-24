import torch

from napl.utils import *
from napl.sim.base import napl_base, hw_params


class dff(napl_base):
    """
    This module is for d flip flop
    """
    def __init__(
            self,
            config={'depth': 1}
        ):
        super().__init__(config, ['depth'], polarity_required=False)
        self.depth = config['depth']
        self.hw = hw_params(pp_delay=self.depth)
        # device-anchor Parameter: tracks device for .to() and lazy buffer init.
        self.reg = torch.nn.Parameter(torch.zeros(1, dtype=self.stype), requires_grad=False)
        # FIFO rows held as a list of tensor references (no per-timestep copy).
        self.buf = None
        self.is_first_call = True
        # circular-buffer index of the oldest stored row (the next one to emit)
        self.head = 0


    def _reset(self):
        self.buf = None
        self.is_first_call = True
        self.head = 0


    def forward(self, input: torch.tensor):
        # input is a spike tensor
        if self.is_first_call:
            zero = torch.zeros_like(input, device=self.reg.device)
            self.buf = [zero.clone() for _ in range(self.depth)]
            self.is_first_call = False

        # FIFO via circular buffer: emit the oldest row and snapshot this
        # timestep's input in that slot without reallocating the full buffer.
        output = self.buf[self.head]
        self.buf[self.head] = input.detach().clone()
        self.head += 1
        if self.head == self.depth:
            self.head = 0
        return output

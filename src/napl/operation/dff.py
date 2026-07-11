import torch

from napl.utils import *
from napl.base import napl_base, hw_params


class dff(napl_base):
    """
    This module is for d flip flop
    """
    def __init__(
            self,
            config={'depth': 1}
        ):
        super().__init__(config, ['depth'], polarity_required=False)
        self.hw = hw_params(pp_delay=1)

        self.depth = config['depth']
        # device-anchor Parameter: tracks device for .to() and lazy buffer init.
        self.reg = torch.nn.Parameter(torch.zeros(1, dtype=self.stype), requires_grad=False)
        # FIFO rows held as a list of tensor references (no per-timestep copy).
        self.buf = None
        self.is_first_call = True
        # circular-buffer index of the oldest stored row (the next one to emit)
        self.head = 0


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.buf = None
        self.is_first_call = True
        self.head = 0


    def forward(self, input: torch.tensor):
        self.tick()
        # input is a spike tensor
        if self.is_first_call:
            zero = torch.zeros_like(input, device=self.reg.device)
            self.buf = [zero.clone() for _ in range(self.depth)]
            self.is_first_call = False

        # FIFO via circular buffer of references: emit the oldest row and store
        # the input by reference in that slot. Rotating references avoids both
        # torch.roll's full-buffer reallocation and the per-timestep clone of the
        # original in-place buffer (the producer emits a fresh tensor each call).
        output = self.buf[self.head]
        self.buf[self.head] = input
        self.head += 1
        if self.head == self.depth:
            self.head = 0
        return output


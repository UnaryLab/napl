import torch
from collections import deque

from napl.utils import *
from napl.base import napl_base, hw_params


class shiftreg(napl_base):
    """
    This module is for shift register
    """
    def __init__(
            self,
            config={'depth': 1}
        ):
        super().__init__(config, ['depth'], polarity_required=False)

        self.depth = config['depth']
        # output is reg[head], which is depth cycles old: latency == depth. RTL
        # i_rst_n must reproduce the i%2 reset pattern below, not all-zeros.
        self.hw = hw_params(pp_delay=self.depth)
        self.reg = torch.nn.Parameter(torch.zeros(self.depth, dtype=self.stype), requires_grad=False)
        for i in range(self.depth):
            self.reg[i].fill_(i%2)
        self.is_first_call = True
        # runtime FIFO of row tensors; built lazily from self.reg on first forward
        self.fifo = None


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.reg.data = torch.zeros(self.depth, dtype=self.stype, device=self.reg.device)
        for i in range(self.depth):
            self.reg[i].fill_(i%2)
        self.is_first_call = True
        self.fifo = None


    def forward(self, input: torch.tensor):
        self.tick()
        # input is a spike tensor
        if self.is_first_call:
            input_shape = list(input.shape)
            input_shape.insert(0, self.depth)
            self.reg.data = torch.zeros(input_shape, dtype=self.stype, device=self.reg.device)
            for i in range(self.depth):
                self.reg[i].fill_(i%2)
            # FIFO over row references: oldest at the left. Storing/popping tensor
            # references makes each timestep a pure pointer op (no per-step clone or
            # indexed copy); input is a fresh tensor each call, so no aliasing risk.
            self.fifo = deque(self.reg.data[i] for i in range(self.depth))
            self.is_first_call = False

        # Read oldest (depth-cycles-old) value, push the new input at the tail.
        output = self.fifo.popleft()
        self.fifo.append(input)
        return output


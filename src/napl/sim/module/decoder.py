import torch
import math

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger


class decoder(napl_base):
    def __init__(
            self,
            config:dict={
                'polarity': 'bipolar',
                'timestep': 256,
                }
        ):
        super().__init__(config, ['polarity', 'timestep'], polarity_required=True)

        # initialize timestep and spike count
        self.timestep = config['timestep']
        self.width = math.ceil(math.log2(self.timestep))

        self.spike_count = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)
        self.spike_value = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)


    def _reset(self):
        self.spike_count.data = torch.zeros(1, dtype=self.ntype, device=self.spike_count.device)
        self.spike_value.data = torch.zeros(1, dtype=self.ntype, device=self.spike_value.device)


    def forward(self, spike: torch.Tensor):
        # get the spike value at the current timestep
        assert self.timestep_cur <= self.timestep, \
            logger.error(f'Timestep <{self.timestep_cur}> exceeds the maximum timestep <{self.timestep}>.')
        # accumulate directly on the (int8) spike: add/add_ type-promote the operand into
        # the ntype accumulator, so an explicit full-size .type(ntype) cast per timestep is
        # redundant. add_ into a float32 destination from an int8 operand is exact and safe.
        # in-place accumulate once the accumulator matches the stream shape;
        # the first timestep must broadcast-expand the (1,) init, which add_ cannot do
        if self.spike_count.shape == spike.shape:
            self.spike_count.add_(spike)
        else:
            self.spike_count.data = self.spike_count.add(spike)
        if self.polarity == 'bipolar':
            # (count/t)*2 - 1, fused to one new tensor + in-place scale/shift
            self.spike_value.data = self.spike_count.div(self.timestep_cur).mul_(2).sub_(1)
        else:
            self.spike_value.data = self.spike_count.div(self.timestep_cur)
        return self.spike_value

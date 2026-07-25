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


    def _reset(self):
        self.spike_count.data = torch.zeros(1, dtype=self.ntype, device=self.spike_count.device)


    @property
    def spike_value(self):
        """
        Progressive value, computed on access from the accumulated spike_count.
        Returns a fresh tensor; before any forward() it is the zeros seed.
        """
        if self.timestep_cur == 0:
            return torch.zeros_like(self.spike_count)
        # sv is the fresh div result, so the in-place bipolar rescale leaves spike_count untouched.
        sv = self.spike_count.div(self.timestep_cur)
        if self.polarity == 'bipolar':
            sv.mul_(2).sub_(1)
        return sv


    def forward(self, spike: torch.Tensor):
        # get the spike value at the current timestep
        assert self.timestep_cur <= self.timestep, \
            logger.error(f'Timestep <{self.timestep_cur}> exceeds the maximum timestep <{self.timestep}>.')
        # float accumulator avoids overflow; the 0/1 spike promotes exactly, so no cast.
        sc = self.spike_count
        # shape-guarded: first forward broadcasts the (1,) seed up to spike's shape
        # out-of-place; steady state accumulates in place to drop a per-timestep alloc.
        if sc.shape == spike.shape:
            sc.add_(spike)
        else:
            sc.data = sc.add(spike)
        # no return: evaluating the spike_value property here would redo the div
        # every timestep; readers access .spike_value on demand instead.

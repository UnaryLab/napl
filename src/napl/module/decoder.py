import torch, math

from napl.utils import *
from napl.base import napl_base
from loguru import logger


class decoder(napl_base):
    def __init__(
            self,
            config:dict={
                'mode': 'bipolar',
                'timestep': 256,
                }
        ):
        super().__init__()

        # check config
        check_config(config, ['mode', 'timestep'])
        self.mode = check_mode(config)
        self.name = check_name(config)

        # initialize timestep and spike count
        self.timestep = config['timestep']
        self.bitwidth = math.ceil(math.log2(self.timestep))
        
        self.timestep_cur = 0
        self.spike_count = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)
        self.spike_value = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)


    def reset(self):
        self.timestep_cur = 0
        self.spike_count.data = torch.zeros(1, dtype=self.ntype)
        self.spike_value.data = torch.zeros(1, dtype=self.ntype)
    
    
    def forward(self, spike: torch.Tensor):
        # get the spike value at the current timestep
        self.timestep_cur += 1
        assert self.timestep_cur <= self.timestep, \
            logger.error(f'Timestep <{self.timestep_cur}> exceeds the maximum timestep <{self.timestep}>.')
        self.spike_count.data = self.spike_count.add(spike.type(self.ntype))
        self.spike_value.data = self.spike_count.div(self.timestep_cur)
        if self.mode == 'bipolar':
            self.spike_value.data = self.spike_value.mul(2).sub(1)
        return self.spike_value
    

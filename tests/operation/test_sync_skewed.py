import math
import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import sync_skewed
from napl.sim.metric import accuracy


class napl_sync_skewed(napl_base):
    def __init__(self, codec_config1, codec_config2, sync_skewed_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder0 = decoder(codec_config1)
        self.decoder1 = decoder(codec_config1)
        self.sync_skewed = sync_skewed(sync_skewed_config)
        self.accuracy0 = accuracy(codec_config1)
        self.accuracy1 = accuracy(codec_config1)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike0, o_spike1 = self.sync_skewed(i_spike0, i_spike1)
        self.decoder0(o_spike0)
        self.decoder1(o_spike1)
        self.accuracy0(o_spike0)
        self.accuracy1(o_spike1)

    
def test_sync_skewed():
    """
    Test sync_skewed with a simple configuration.
    """

    codec_config1={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 3,
    }
    sync_skewed_config={
        'width' : 3,
    }

    # Generate random inputs based on polarity
    input_0_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)
    input_mask = input_0_cpu < input_1_cpu
    input_0_new = torch.where(input_mask, input_0_cpu, input_1_cpu)
    input_1_new = torch.where(~input_mask, input_0_cpu, input_1_cpu)
    # make sure divisor is not 0
    input_1_new = torch.where(input_1_new==0, 1, input_1_new)
    input_0_cpu = input_0_new
    input_1_cpu = input_1_new

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        sync_skewed_inst = napl_sync_skewed(codec_config1, codec_config2, sync_skewed_config).to(device)
        sync(device)
        start = time.perf_counter()
        sync_skewed_inst(input_0, input_1, timesteps=codec_config1['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        sync_skewed_inst.accuracy0.analyze(input_0, verbose=True)
        sync_skewed_inst.accuracy1.analyze(input_1, verbose=True)
        assert sync_skewed_inst.sync_skewed.timestep_cur == codec_config1['timestep']
        sync_skewed_inst.reset()
        assert sync_skewed_inst.sync_skewed.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')
    
    print('Test passed.')


if __name__ == '__main__':
    test_sync_skewed()

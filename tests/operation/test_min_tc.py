import math
import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import min_tc
from napl.sim.metric import accuracy


class napl_min_tc(napl_base):
    def __init__(self, codec_config1, codec_config2, min_tc_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder = decoder(codec_config1)
        self.min_tc = min_tc(min_tc_config)
        self.accuracy = accuracy({'polarity': codec_config1['polarity']})


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.min_tc(i_spike0, i_spike1)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def test_min_tc():
    """
    Test min_tc with a simple configuration.
    """

    codec_config1={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'temporal',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'temporal',
        'dim': 2,
    }
    min_tc_config=codec_config1

    # Generate random inputs based on polarity
    input_0_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        min_tc_inst = napl_min_tc(codec_config1, codec_config2, min_tc_config).to(device)
        sync(device)
        start = time.perf_counter()
        min_tc_inst(input_0, input_1, timesteps=codec_config1['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = torch.min(input_0, input_1)
        min_tc_inst.accuracy.analyze(r_value, verbose=True)
        assert min_tc_inst.min_tc.timestep_cur == codec_config1['timestep']
        min_tc_inst.reset()
        assert min_tc_inst.min_tc.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')
    
    print('Test passed.')


if __name__ == '__main__':
    test_min_tc()

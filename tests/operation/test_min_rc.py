import math
import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import min_rc
from napl.sim.metric import accuracy


class napl_min_rc(napl_base):
    def __init__(self, codec_config1, codec_config2, codec_config3, min_rc_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder0 = decoder(codec_config1)
        self.decoder1 = decoder(codec_config3)
        self.min_rc = min_rc(min_rc_config)
        self.accuracy0 = accuracy({'polarity': codec_config1['polarity']})
        self.accuracy1 = accuracy({'polarity': codec_config3['polarity']})


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike0, o_spike1 = self.min_rc(i_spike0, i_spike1)
        self.decoder0(o_spike0)
        self.decoder1(o_spike1)
        self.accuracy0(o_spike0)
        self.accuracy1(o_spike1)

    
def test_min_rc():
    """
    Test min_rc with a simple configuration.
    """

    codec_config1={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 2,
    }
    codec_config3={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 2,
    }
    min_rc_config=codec_config1

    # Generate random inputs based on polarity
    input_0_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        min_rc_inst = napl_min_rc(codec_config1, codec_config2, codec_config3, min_rc_config).to(device)
        sync(device)
        start = time.perf_counter()
        min_rc_inst(input_0, input_1, timesteps=codec_config1['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = torch.min(input_0, input_1)
        r_value_arg = torch.argmin(torch.stack([input_0, input_1], dim=0), dim=0)
        _, value_idx = min_rc_inst.accuracy0.analyze(r_value, verbose=True)
        _, arg_idx = min_rc_inst.accuracy1.analyze(r_value_arg, verbose=True)

        print(f'[{device}] value max error index: {value_idx.item():7d}; arg max error index: {arg_idx.item():7d}; time: {elapsed * 1000:.1f} ms')
        assert min_rc_inst.min_rc.timestep_cur == codec_config1['timestep']
        min_rc_inst.reset()
        assert min_rc_inst.min_rc.timestep_cur == 0
    
    print('Test passed.')


if __name__ == '__main__':
    test_min_rc()

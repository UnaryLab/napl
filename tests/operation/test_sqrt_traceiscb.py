import math
import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import sqrt_traceiscb
from napl.sim.metric import accuracy


class napl_sqrt_traceiscb(napl_base):
    def __init__(self, codec_config, sqrt_traceiscb_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.sqrt_traceiscb = sqrt_traceiscb(sqrt_traceiscb_config)
        self.accuracy = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.sqrt_traceiscb(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def test_sqrt_traceiscb():
    """
    Test sqrt_traceiscb with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    sqrt_traceiscb_config={
        'polarity': 'bipolar',
    }
    
    # Generate random inputs based on polarity, ensure positive numbers
    input_cpu = gen_rand_tensor('unipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)

    # generate the napl_sqrt_traceiscb instance
    for device in devices():
        input = input_cpu.to(device)
        sqrt_traceiscb_inst = napl_sqrt_traceiscb(codec_config, sqrt_traceiscb_config).to(device)
        sync(device)
        start = time.perf_counter()
        sqrt_traceiscb_inst(input, timesteps=codec_config['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = torch.sqrt(input)
        sqrt_traceiscb_inst.accuracy.analyze(r_value, verbose=True)
        assert sqrt_traceiscb_inst.sqrt_traceiscb.timestep_cur == codec_config['timestep']
        sqrt_traceiscb_inst.reset()
        assert sqrt_traceiscb_inst.sqrt_traceiscb.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')

    print('Test passed.')


if __name__ == '__main__':
    test_sqrt_traceiscb()

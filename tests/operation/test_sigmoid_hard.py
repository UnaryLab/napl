import math
import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import sigmoid_hard
from napl.sim.metric import accuracy


class napl_sigmoid_hard(napl_base):
    def __init__(self, codec_config, sigmoid_hard_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.sigmoid_hard = sigmoid_hard(sigmoid_hard_config)
        self.accuracy = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.sigmoid_hard(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def test_sigmoid_hard():
    """
    Test sigmoid_hard with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    sigmoid_hard_config={
        'polarity': 'bipolar',
    }
    
    # Generate random inputs based on polarity, ensure positive numbers
    input_cpu = gen_rand_tensor('bipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)
    # input = gen_arange_tensor('unipolar', width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_sigmoid_hard instance
    for device in devices():
        input = input_cpu.to(device)
        sigmoid_hard_inst = napl_sigmoid_hard(codec_config, sigmoid_hard_config).to(device)
        sync(device)
        start = time.perf_counter()
        sigmoid_hard_inst(input, timesteps=codec_config['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = torch.nn.Hardsigmoid()(input * 3)
        sigmoid_hard_inst.accuracy.analyze(r_value, verbose=True)
        assert sigmoid_hard_inst.sigmoid_hard.timestep_cur == codec_config['timestep']
        sigmoid_hard_inst.reset()
        assert sigmoid_hard_inst.sigmoid_hard.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')

    print('Test passed.')


if __name__ == '__main__':
    test_sigmoid_hard()

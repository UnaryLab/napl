import math
import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import relu_sat
from napl.sim.metric import accuracy


class napl_relu_sat(napl_base):
    def __init__(self, codec_config, relu_sat_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.relu_sat = relu_sat(relu_sat_config)
        self.accuracy = accuracy({'polarity': codec_config['polarity']})


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.relu_sat(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def test_relu_sat():
    """
    Test relu_sat with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    relu_sat_config={}
    
    # Generate random inputs based on polarity, ensure positive numbers
    input_cpu = gen_rand_tensor('bipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)
    # input = gen_arange_tensor('unipolar', width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_relu_sat instance
    for device in devices():
        input = input_cpu.to(device)
        relu_sat_inst = napl_relu_sat(codec_config, relu_sat_config).to(device)
        sync(device)
        start = time.perf_counter()
        relu_sat_inst(input, timesteps=codec_config['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = torch.nn.ReLU()(input)
        relu_sat_inst.accuracy.analyze(r_value, verbose=True)
        assert relu_sat_inst.relu_sat.timestep_cur == codec_config['timestep']
        relu_sat_inst.reset()
        assert relu_sat_inst.relu_sat.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')

    print('Test passed.')


if __name__ == '__main__':
    test_relu_sat()

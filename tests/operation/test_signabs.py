import math
import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import signabs
from napl.sim.metric import accuracy


class napl_signabs(napl_base):
    def __init__(self, codec_config, signabs_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder_sign = decoder(codec_config)
        self.decoder_abs = decoder(codec_config)
        self.signabs = signabs(signabs_config)
        self.accuracy_sign = accuracy(codec_config)
        self.accuracy_abs = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike_sign, o_spike_abs = self.signabs(i_spike)
        self.decoder_sign(o_spike_sign)
        self.decoder_abs(o_spike_abs)
        self.accuracy_sign(o_spike_sign)
        self.accuracy_abs(o_spike_abs)


def test_signabs():
    """
    Test signabs with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    signabs_config={
        'width': 3
    }

    # Generate random inputs based on polarity
    input_cpu = gen_rand_tensor(codec_config['polarity'], shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        signabs_inst = napl_signabs(codec_config, signabs_config).to(device)
        sync(device)
        start = time.perf_counter()
        signabs_inst(input, timesteps=codec_config['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value_sign = -torch.sign(input)
        r_value_abs = torch.abs(input)
        signabs_inst.accuracy_sign.analyze(r_value_sign, verbose=True)
        signabs_inst.accuracy_abs.analyze(r_value_abs, verbose=True)
        assert signabs_inst.signabs.timestep_cur == codec_config['timestep']
        signabs_inst.reset()
        assert signabs_inst.signabs.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')

    print('Test passed.')


if __name__ == '__main__':
    test_signabs()

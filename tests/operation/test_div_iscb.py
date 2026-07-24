import math
import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import div_iscb
from napl.sim.metric import accuracy


class napl_div_iscb(napl_base):
    def __init__(self, codec_config1, codec_config2, div_iscb_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder = decoder(codec_config1)
        self.accuracy = accuracy({'polarity': codec_config1['polarity']})
        self.div_iscb = div_iscb(div_iscb_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.div_iscb(i_spike0, i_spike1)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def test_div_iscb():
    """
    Test div_iscb with a simple configuration.
    """

    polarity = 'bipolar'
    codec_config1={
        'polarity': polarity,
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': polarity,
        'timestep': 256,
        'generator': 'sobol',
        'dim': 2,
    }
    div_iscb_config={
        'polarity': polarity,
    }

    # Generate random inputs based on polarity
    input_0 = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1 = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)
    input_mask = torch.abs(input_0) < torch.abs(input_1)
    input_0_new = torch.where(input_mask, input_0, input_1)
    input_1_new = torch.where(~input_mask, input_0, input_1)
    # make sure divisor is not 0
    input_1_new = torch.where(input_1_new==0, 1, input_1_new)
    input_0 = input_0_new
    input_1 = input_1_new

    for device in devices():
        dividend = input_0.to(device)
        divisor = input_1.to(device)
        div_iscb_inst = napl_div_iscb(codec_config1, codec_config2, div_iscb_config).to(device)

        sync(device)
        start = time.perf_counter()
        div_iscb_inst(dividend, divisor, timesteps=codec_config1['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = dividend / divisor
        error, _ = div_iscb_inst.accuracy.analyze(r_value, verbose=True)
        rmse = error.pow(2).mean().sqrt().item()
        assert rmse < 0.25, f'[{device}] rmse={rmse:.4f}'

        assert div_iscb_inst.div_iscb.timestep_cur == codec_config1['timestep']
        div_iscb_inst.reset()
        assert div_iscb_inst.div_iscb.timestep_cur == 0
        print(f'[{device}] rmse={rmse:.4f}, time={elapsed:.3f}s')
    
    print('Test passed.')


if __name__ == '__main__':
    test_div_iscb()

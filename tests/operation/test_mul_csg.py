import math
import time

import torch


from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import mul_csg
from napl.sim.metric import accuracy


class napl_mul_csg(napl_base):
    def __init__(self, codec_config, mul_csg_config):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.mul_csg = mul_csg(mul_csg_config)
        self.accuracy = accuracy({'polarity': codec_config['polarity']})


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike = self.encoder(input_0)
        o_spike = self.mul_csg(i_spike, input_1)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def test_mul_csg_rank2():
    """Verify mul_csg preserves rank-two spike tensors when multiplied by one."""
    config = {
        'polarity': 'bipolar',
        'timestep': 4,
        'generator': 'sobol',
    }
    input_0_cpu = torch.tensor(
        [[0, 1, 0], [1, 0, 1]],
        dtype=global_config.stype,
    )
    input_1_cpu = torch.ones((2, 3), dtype=global_config.ntype)

    for device in devices():
        mul_csg_inst = mul_csg(config).to(device)
        result = mul_csg_inst(
            input_0_cpu.to(device),
            input_1_cpu.to(device),
        )

        assert result.ndim == 2
        assert torch.equal(result.cpu(), input_0_cpu)

    
def test_mul_csg():
    """Verify mul_csg tracks real multiplication within stochastic tolerance across devices."""

    codec_config={
        'polarity': 'bipolar',
        'timestep': 1024,
        'generator': 'sobol',
    }
    mul_csg_config=codec_config

    input_0_cpu = gen_rand_tensor(codec_config['polarity'], shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config['polarity'], shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        mul_csg_inst = napl_mul_csg(codec_config, mul_csg_config).to(device)
        sync(device)
        start = time.perf_counter()
        mul_csg_inst(input_0, input_1, timesteps=codec_config['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = input_0 * input_1
        mul_csg_inst.accuracy.analyze(r_value, verbose=True)
        assert mul_csg_inst.mul_csg.timestep_cur == codec_config['timestep']
        mul_csg_inst.reset()
        assert mul_csg_inst.mul_csg.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')

    print('Test passed.')


if __name__ == '__main__':
    test_mul_csg_rank2()
    test_mul_csg()

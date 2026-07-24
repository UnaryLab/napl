import math
import time


from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import gt_rc
from napl.sim.metric import accuracy


class napl_gt_rc(napl_base):
    def __init__(self, codec_config1, codec_config2, codec_config3, gt_rc_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder = decoder(codec_config3)
        self.accuracy = accuracy({'polarity': codec_config3['polarity']})
        self.gt_rc = gt_rc(gt_rc_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.gt_rc(i_spike0, i_spike1)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def test_gt_rc():
    """
    Test gt_rc with a simple configuration.
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
    gt_rc_config=codec_config1

    # Generate random inputs based on polarity
    input_0_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        gt_rc_inst = napl_gt_rc(codec_config1, codec_config2, codec_config3, gt_rc_config).to(device)

        sync(device)
        start = time.perf_counter()
        gt_rc_inst(input_0, input_1, timesteps=codec_config1['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = (input_0 > input_1).type(global_config.ntype)
        error, idx = gt_rc_inst.accuracy.analyze(r_value, verbose=True)
        rmse = error.pow(2).mean().sqrt().item()
        bound = 2.0 / math.sqrt(codec_config1['timestep'])
        assert rmse < bound, f'[{device}] rmse={rmse:.4f}, bound={bound:.4f}'

        print(f'[{device}] rmse={rmse:.4f}, bound={bound:.4f}, time={elapsed:.3f}s, '
              f'max-error index={idx.item():7d}')
        assert gt_rc_inst.gt_rc.timestep_cur == codec_config1['timestep']
        gt_rc_inst.reset()
        assert gt_rc_inst.gt_rc.timestep_cur == 0
    
    print('Test passed.')


if __name__ == '__main__':
    test_gt_rc()

import math
import time


from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import devices, gen_rand_tensor, sync
from napl.module import encoder, decoder
from napl.operation import lt_rc
from napl.metric import analyze_error


class napl_lt_rc(napl_base):
    def __init__(self, codec_config1, codec_config2, codec_config3, lt_rc_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder = decoder(codec_config3)
        self.lt_rc = lt_rc(lt_rc_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.lt_rc(i_spike0, i_spike1)
        self.decoder(o_spike)

    
def test_lt_rc():
    """
    Test lt_rc with a simple configuration.
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
    lt_rc_config=codec_config1

    # Generate random inputs based on polarity
    input_0_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        lt_rc_inst = napl_lt_rc(codec_config1, codec_config2, codec_config3, lt_rc_config).to(device)

        sync(device)
        start = time.perf_counter()
        lt_rc_inst(input_0, input_1, timesteps=codec_config1['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = (input_0 < input_1).type(global_config.ntype)
        error, idx = analyze_error(lt_rc_inst.decoder.spike_value, r_value)
        rmse = error.pow(2).mean().sqrt().item()
        bound = 2.0 / math.sqrt(codec_config1['timestep'])
        assert rmse < bound, f'[{device}] rmse={rmse:.4f}, bound={bound:.4f}'

        print(f'[{device}] rmse={rmse:.4f}, bound={bound:.4f}, time={elapsed:.3f}s, '
              f'max-error index={idx.item():7d}')
        assert lt_rc_inst.lt_rc.timestep_cur == codec_config1['timestep']
        lt_rc_inst.reset()
        assert lt_rc_inst.lt_rc.timestep_cur == 0
    
    print('Test passed.')


if __name__ == '__main__':
    test_lt_rc()

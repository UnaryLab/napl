import math
import time


from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import devices, gen_rand_tensor, sync
from napl.module import encoder, decoder
from napl.operation import bi2uni
from napl.metric import analyze_error


class napl_bi2uni(napl_base):
    def __init__(self, codec_config1, codec_config2, bi2uni_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config1)
        self.decoder = decoder(codec_config2)
        self.bi2uni = bi2uni(bi2uni_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.bi2uni(i_spike)
        self.decoder(o_spike)

    
def test_bi2uni():
    """
    Test bi2uni with a simple configuration.
    """

    codec_config1={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    bi2uni_config={
        'width': 2,
    }
    
    # Generate random inputs based on polarity
    # all inputs shall be positive
    input_cpu = gen_rand_tensor(
        codec_config2['polarity'],
        shape=(10000,),
        width=math.log2(codec_config1['timestep']),
    ).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        bi2uni_inst = napl_bi2uni(codec_config1, codec_config2, bi2uni_config).to(device)

        sync(device)
        start = time.perf_counter()
        bi2uni_inst(input, timesteps=codec_config1['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        error, _ = analyze_error(bi2uni_inst.decoder.spike_value, input)
        rmse = error.pow(2).mean().sqrt().item()
        bound = 2.0 / math.sqrt(codec_config1['timestep'])
        assert rmse < bound, f'[{device}] rmse={rmse:.4f}, bound={bound:.4f}'

        assert bi2uni_inst.bi2uni.timestep_cur == codec_config1['timestep']
        bi2uni_inst.reset()
        assert bi2uni_inst.bi2uni.timestep_cur == 0
        print(f'[{device}] rmse={rmse:.4f}, time={elapsed:.3f}s')

    print('Test passed.')


if __name__ == '__main__':
    test_bi2uni()

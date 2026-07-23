import math
import time


from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import devices, gen_rand_tensor, sync
from napl.module import encoder, decoder
from napl.operation import mul_csg
from napl.metric import analyze_error


class napl_mul_csg(napl_base):
    def __init__(self, codec_config, mul_csg_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.mul_csg = mul_csg(mul_csg_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input_0)
        o_spike = self.mul_csg(i_spike, input_1)
        self.decoder(o_spike)

    
def test_mul_csg():
    """
    Test mul_csg with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 1024,
        'generator': 'sobol',
    }
    mul_csg_config=codec_config

    # Generate random inputs based on polarity
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
        analyze_error(mul_csg_inst.decoder.spike_value, r_value)
        assert mul_csg_inst.mul_csg.timestep_cur == codec_config['timestep']
        mul_csg_inst.reset()
        assert mul_csg_inst.mul_csg.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')

    print('Test passed.')


if __name__ == '__main__':
    test_mul_csg()

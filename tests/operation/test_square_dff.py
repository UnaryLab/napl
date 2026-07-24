import math
import time


from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import square_dff
from napl.sim.metric import accuracy


class napl_square_dff(napl_base):
    def __init__(self, codec_config, square_dff_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.square_dff = square_dff(square_dff_config)
        self.accuracy = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.square_dff(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def test_square_dff():
    """
    Test square_dff with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    square_dff_config={
        'polarity': 'bipolar',
        'depth': 1
    }
    
    # Generate random inputs based on polarity
    input_cpu = gen_rand_tensor(codec_config['polarity'], shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)

    # generate the napl_square_dff instance
    for device in devices():
        input = input_cpu.to(device)
        square_dff_inst = napl_square_dff(codec_config, square_dff_config).to(device)
        sync(device)
        start = time.perf_counter()
        square_dff_inst(input, timesteps=codec_config['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = input * input
        square_dff_inst.accuracy.analyze(r_value, verbose=True)
        assert square_dff_inst.square_dff.timestep_cur == codec_config['timestep']
        square_dff_inst.reset()
        assert square_dff_inst.square_dff.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')

    print('Test passed.')


if __name__ == '__main__':
    test_square_dff()

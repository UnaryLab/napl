import math
import time


from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import uni2bi
from napl.sim.metric import accuracy


class napl_uni2bi(napl_base):
    def __init__(self, codec_config1, codec_config2, uni2bi_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config1)
        self.decoder = decoder(codec_config2)
        self.uni2bi = uni2bi(uni2bi_config)
        self.accuracy = accuracy(codec_config2)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.uni2bi(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def test_uni2bi():
    """
    Test uni2bi with a simple configuration.
    """

    codec_config1={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    uni2bi_config={
        'width': 3,
    }
    
    # Generate random inputs based on polarity
    input_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)

    # generate the napl_uni2bi instance
    for device in devices():
        input = input_cpu.to(device)
        uni2bi_inst = napl_uni2bi(codec_config1, codec_config2, uni2bi_config).to(device)
        sync(device)
        start = time.perf_counter()
        uni2bi_inst(input, timesteps=codec_config1['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        uni2bi_inst.accuracy.analyze(input, verbose=True)
        assert uni2bi_inst.uni2bi.timestep_cur == codec_config1['timestep']
        uni2bi_inst.reset()
        assert uni2bi_inst.uni2bi.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')

    print('Test passed.')


if __name__ == '__main__':
    test_uni2bi()

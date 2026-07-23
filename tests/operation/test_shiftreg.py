import math
import time


from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import devices, gen_rand_tensor, sync
from napl.module import encoder, decoder
from napl.operation import shiftreg
from napl.metric import analyze_error


class napl_shiftreg(napl_base):
    def __init__(self, codec_config, shiftreg_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.shiftreg = shiftreg(shiftreg_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.shiftreg(i_spike)
        self.decoder(o_spike)

    
def test_shiftreg():
    """
    Test shiftreg with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    shiftreg_config={
        'depth': 2
    }
    
    # Generate random inputs based on polarity
    input_cpu = gen_rand_tensor(codec_config['polarity'], shape=(10,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)

    # generate the napl_shiftreg instance
    for device in devices():
        input = input_cpu.to(device)
        shiftreg_inst = napl_shiftreg(codec_config, shiftreg_config).to(device)
        sync(device)
        start = time.perf_counter()
        shiftreg_inst(input, timesteps=codec_config['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        analyze_error(shiftreg_inst.decoder.spike_value, input)
        assert shiftreg_inst.shiftreg.timestep_cur == codec_config['timestep']
        shiftreg_inst.reset()
        assert shiftreg_inst.shiftreg.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')
    
    print('Test passed.')


if __name__ == '__main__':
    test_shiftreg()

import math
import time


from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import devices, gen_rand_tensor, sync
from napl.module import encoder, decoder
from napl.operation import dff
from napl.metric import analyze_error


class napl_dff(napl_base):
    def __init__(self, codec_config, dff_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.dff = dff(dff_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.dff(i_spike)
        self.decoder(o_spike)

    
def test_dff():
    """
    Test dff with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    dff_config={
        'depth': 1
    }
    
    # Generate random inputs based on polarity
    input_cpu = gen_rand_tensor(
        codec_config['polarity'],
        shape=(10,),
        width=math.log2(codec_config['timestep']),
    ).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        dff_inst = napl_dff(codec_config, dff_config).to(device)

        sync(device)
        start = time.perf_counter()
        dff_inst(input, timesteps=codec_config['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        error, _ = analyze_error(dff_inst.decoder.spike_value, input)
        rmse = error.pow(2).mean().sqrt().item()
        bound = 2.0 / math.sqrt(codec_config['timestep'])
        assert rmse < bound, f'[{device}] rmse={rmse:.4f}, bound={bound:.4f}'

        assert dff_inst.dff.timestep_cur == codec_config['timestep']
        dff_inst.reset()
        assert dff_inst.dff.timestep_cur == 0
        print(f'[{device}] rmse={rmse:.4f}, time={elapsed:.3f}s')
    
    print('Test passed.')


if __name__ == '__main__':
    test_dff()

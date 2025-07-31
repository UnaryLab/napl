import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import sqrt_emit
from napl.metric import report_error


class napl_sqrt_emit(napl_base):
    def __init__(self, codec_config, sqrt_emit_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.sqrt_emit = sqrt_emit(sqrt_emit_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.sqrt_emit(i_spike)
        self.decoder(o_spike)

    
def test_sqrt_emit():
    """
    Test sqrt_emit with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    codec_config={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 4,
    }
    sqrt_emit_config={
        'polarity': 'unipolar',
    }
    
    # Generate random inputs based on polarity, ensure positive numbers
    input = gen_rand_tensor('unipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)
    # input = gen_arange_tensor('unipolar', width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_sqrt_emit instance
    sqrt_emit_inst = napl_sqrt_emit(codec_config, sqrt_emit_config).to(device)
    sqrt_emit_inst(input, timesteps=codec_config['timestep'])

    # calculate the reference output
    r_value = torch.sqrt(input)

    # report the error
    report_error(sqrt_emit_inst.decoder.spike_value, r_value)

    assert sqrt_emit_inst.sqrt_emit.timestep_cur == codec_config['timestep']
    sqrt_emit_inst.reset()

    print('Test passed.')


if __name__ == '__main__':
    test_sqrt_emit()


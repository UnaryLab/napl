import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import square
from napl.metric import report_error


class napl_square(napl_base):
    def __init__(self, codec_config, square_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.square = square(square_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.square(i_spike)
        self.decoder(o_spike)

    
def test_square():
    """
    Test square with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config={
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    square_config={
        'mode': 'bipolar',
        'delay': 1
    }
    
    # Generate random inputs based on mode
    input = gen_rand_tensor(codec_config['mode'], shape=(1000,), bitwidth=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_square instance
    square_inst = napl_square(codec_config, square_config).to(device)
    square_inst(input, timesteps=codec_config['timestep'])

    # calculate the reference output
    r_value = input * input

    # report the error
    report_error(square_inst.decoder.spike_value, r_value)

    print('Test passed.')


if __name__ == '__main__':
    test_square()


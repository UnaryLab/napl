import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import sqrt_insert
from napl.metric import report_error


class napl_sqrt_insert(napl_base):
    def __init__(self, codec_config, sqrt_insert_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.sqrt_insert = sqrt_insert(sqrt_insert_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.sqrt_insert(i_spike)
        self.decoder(o_spike)

    
def test_sqrt_insert():
    """
    Test sqrt_insert with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config={
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    sqrt_insert_config={
        'mode': 'bipolar',
        'delay': 1
    }
    
    # Generate random inputs based on mode
    input = gen_rand_tensor(codec_config['mode'], shape=(1000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_sqrt_insert instance
    sqrt_insert_inst = napl_sqrt_insert(codec_config, sqrt_insert_config).to(device)
    sqrt_insert_inst(input, timesteps=codec_config['timestep'])

    # calculate the reference output
    r_value = input * input

    # report the error
    report_error(sqrt_insert_inst.decoder.spike_value, r_value)

    sqrt_insert_inst.reset()

    print('Test passed.')


if __name__ == '__main__':
    test_sqrt_insert()


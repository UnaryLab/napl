import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import bi2uni
from napl.metric import report_error


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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config1={
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'mode': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    bi2uni_config={
        'bitwidth': 3,
    }
    
    # Generate random inputs based on mode
    # all inputs shall be positive
    input = gen_rand_tensor(codec_config2['mode'], shape=(1000,), bitwidth=math.log2(codec_config1['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_bi2uni instance
    bi2uni_inst = napl_bi2uni(codec_config1, codec_config2, bi2uni_config).to(device)
    bi2uni_inst(input, timesteps=codec_config1['timestep'])

    # calculate the reference output
    r_value = input

    # report the error
    report_error(bi2uni_inst.decoder.spike_value, r_value)

    bi2uni_inst.reset()

    print('Test passed.')


if __name__ == '__main__':
    test_bi2uni()


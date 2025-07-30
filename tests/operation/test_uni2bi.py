import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import uni2bi
from napl.metric import report_error


class napl_uni2bi(napl_base):
    def __init__(self, codec_config1, codec_config2, uni2bi_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config1)
        self.decoder = decoder(codec_config2)
        self.uni2bi = uni2bi(uni2bi_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.uni2bi(i_spike)
        self.decoder(o_spike)

    
def test_uni2bi():
    """
    Test uni2bi with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config1={
        'mode': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    uni2bi_config={
        'width': 3,
    }
    
    # Generate random inputs based on mode
    input = gen_rand_tensor(codec_config1['mode'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_uni2bi instance
    uni2bi_inst = napl_uni2bi(codec_config1, codec_config2, uni2bi_config).to(device)
    uni2bi_inst(input, timesteps=codec_config1['timestep'])

    # calculate the reference output
    r_value = input

    # report the error
    report_error(uni2bi_inst.decoder.spike_value, r_value)

    uni2bi_inst.reset()

    print('Test passed.')


if __name__ == '__main__':
    test_uni2bi()


import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import div_cordiv
from napl.metric import report_error


class napl_div_cordiv(napl_base):
    def __init__(self, codec_config1, codec_config2, div_cordiv_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder = decoder(codec_config1)
        self.div_cordiv = div_cordiv(div_cordiv_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.div_cordiv(i_spike0, i_spike1)
        self.decoder(o_spike)

    
def test_div_cordiv():
    """
    Test div_cordiv with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config1={
        'mode': 'unipolar',
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
    div_cordiv_config={
        'depth' : 2, 
        'generator' : 'Sobol',
    }

    # Generate random inputs based on mode
    # using the same dim for two codec configs to ensure we have correlated spikes to div_cordiv
    input_0 = gen_rand_tensor(codec_config1['mode'], shape=(1000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype).to(device)
    input_1 = gen_rand_tensor(codec_config2['mode'], shape=(1000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype).to(device)
    input_mask = input_0 < input_1
    input_0_new = torch.where(input_mask, input_0, input_1)
    input_1_new = torch.where(~input_mask, input_0, input_1)
    # make sure divisor is not 0
    input_1_new = torch.where(input_1_new==0, 1, input_1_new)
    input_0 = input_0_new
    input_1 = input_1_new

    # generate the napl_div_cordiv instance
    div_cordiv_inst = napl_div_cordiv(codec_config1, codec_config2, div_cordiv_config).to(device)
    div_cordiv_inst(input_0, input_1, timesteps=codec_config1['timestep'])

    # calculate the reference output
    r_value = input_0 / input_1

    # report the error
    report_error(div_cordiv_inst.decoder.spike_value, r_value)

    div_cordiv_inst.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_div_cordiv()


import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import dff
from napl.metric import report_error


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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    dff_config={
        'depth': 1
    }
    
    # Generate random inputs based on polarity
    input = gen_rand_tensor(codec_config['polarity'], shape=(10,), width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_dff instance
    dff_inst = napl_dff(codec_config, dff_config).to(device)
    dff_inst(input, timesteps=codec_config['timestep'])

    # calculate the reference output
    r_value = input

    # report the error
    report_error(dff_inst.decoder.spike_value, r_value)

    assert dff_inst.dff.timestep_cur == codec_config['timestep']
    dff_inst.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_dff()


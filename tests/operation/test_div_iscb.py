import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import div_iscb
from napl.metric import report_error


class napl_div_iscb(napl_base):
    def __init__(self, codec_config1, codec_config2, div_iscb_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder = decoder(codec_config1)
        self.div_iscb = div_iscb(div_iscb_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.div_iscb(i_spike0, i_spike1)
        self.decoder(o_spike)

    
def test_div_iscb():
    """
    Test div_iscb with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    polarity = 'bipolar'
    codec_config1={
        'polarity': polarity,
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': polarity,
        'timestep': 256,
        'generator': 'sobol',
        'dim': 2,
    }
    div_iscb_config={
        'polarity': polarity,
    }

    # Generate random inputs based on polarity
    input_0 = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype).to(device)
    input_1 = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype).to(device)
    input_mask = torch.abs(input_0) < torch.abs(input_1)
    input_0_new = torch.where(input_mask, input_0, input_1)
    input_1_new = torch.where(~input_mask, input_0, input_1)
    # make sure divisor is not 0
    input_1_new = torch.where(input_1_new==0, 1, input_1_new)
    input_0 = input_0_new
    input_1 = input_1_new

    # generate the napl_div_iscb instance
    div_iscb_inst = napl_div_iscb(codec_config1, codec_config2, div_iscb_config).to(device)
    div_iscb_inst(input_0, input_1, timesteps=codec_config1['timestep'])

    # calculate the reference output
    r_value = input_0 / input_1

    # report the error
    report_error(div_iscb_inst.decoder.spike_value, r_value)

    assert div_iscb_inst.div_iscb.timestep_cur == codec_config1['timestep']
    div_iscb_inst.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_div_iscb()


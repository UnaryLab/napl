import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import min_rc
from napl.metric import report_error


class napl_min_rc(napl_base):
    def __init__(self, codec_config1, codec_config2, codec_config3, min_rc_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder0 = decoder(codec_config1)
        self.decoder1 = decoder(codec_config3)
        self.min_rc = min_rc(min_rc_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike0, o_spike1 = self.min_rc(i_spike0, i_spike1)
        self.decoder0(o_spike0)
        self.decoder1(o_spike1)

    
def test_min_rc():
    """
    Test min_rc with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config1={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 2,
    }
    codec_config3={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 2,
    }
    min_rc_config=codec_config1

    # Generate random inputs based on polarity
    input_0 = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype).to(device)
    input_1 = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_min_rc instance
    min_rc_inst = napl_min_rc(codec_config1, codec_config2, codec_config3, min_rc_config).to(device)
    min_rc_inst(input_0, input_1, timesteps=codec_config1['timestep'])

    # calculate the reference output
    r_value = torch.min(input_0, input_1)
    r_value_arg = torch.argmin(torch.stack([input_0, input_1], dim=0), dim=0)

    # report the error
    _, value_idx = report_error(min_rc_inst.decoder0.spike_value, r_value)
    _, arg_idx = report_error(min_rc_inst.decoder1.spike_value, r_value_arg)

    print(f'index with the largest error for value: {value_idx.item():7d}\tinput 0: {input_0[value_idx].item(): .5f}\tinput 1: {input_1[value_idx].item(): .5f}\tvalue min_rc: {min_rc_inst.decoder0.spike_value[value_idx].item(): .5f}')
    print(f'index with the largest error for index: {arg_idx.item():7d}\tinput 0: {input_0[arg_idx].item(): .5f}\tinput 1: {input_1[arg_idx].item(): .5f}\tindex min_rc: {min_rc_inst.decoder1.spike_value[arg_idx].item(): .5f}')
    
    assert min_rc_inst.min_rc.timestep_cur == codec_config1['timestep']
    min_rc_inst.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_min_rc()


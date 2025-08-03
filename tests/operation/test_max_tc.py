import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import max_tc
from napl.metric import report_error


class napl_max_tc(napl_base):
    def __init__(self, codec_config1, codec_config2, max_tc_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder = decoder(codec_config1)
        self.max_tc = max_tc(max_tc_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.max_tc(i_spike0, i_spike1)
        self.decoder(o_spike)

    
def test_max_tc():
    """
    Test max_tc with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config1={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'temporal',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'temporal',
        'dim': 2,
    }
    max_tc_config=codec_config1

    # Generate random inputs based on polarity
    input_0 = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype).to(device)
    input_1 = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_max_tc instance
    max_tc_inst = napl_max_tc(codec_config1, codec_config2, max_tc_config).to(device)
    max_tc_inst(input_0, input_1, timesteps=codec_config1['timestep'])

    # calculate the reference output
    r_value = torch.max(input_0, input_1)

    # report the error
    report_error(max_tc_inst.decoder.spike_value, r_value)

    assert max_tc_inst.max_tc.timestep_cur == codec_config1['timestep']
    max_tc_inst.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_max_tc()


import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import mul_and
from napl.metric import report_error


class napl_mul_and(napl_base):
    def __init__(self, codec_config1, codec_config2, mul_and_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder = decoder(codec_config1)
        self.mul_and = mul_and(mul_and_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.mul_and(i_spike0, i_spike1)
        self.decoder(o_spike)

    
def test_mul_and():
    """
    Test mul_and with a simple configuration.
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
    mul_and_config=codec_config1

    # Generate random inputs based on polarity
    input_0 = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype).to(device)
    input_1 = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_mul_and instance
    mul_and_inst = napl_mul_and(codec_config1, codec_config2, mul_and_config).to(device)
    mul_and_inst(input_0, input_1, timesteps=codec_config1['timestep'])

    # calculate the reference output
    r_value = input_0 * input_1

    # report the error
    report_error(mul_and_inst.decoder.spike_value, r_value)

    assert mul_and_inst.mul_and.timestep_cur == codec_config1['timestep']
    mul_and_inst.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_mul_and()


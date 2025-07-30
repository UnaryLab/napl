import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import mul_csg
from napl.metric import report_error


class napl_mul_csg(napl_base):
    def __init__(self, codec_config, mul_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.mul = mul_csg(mul_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input_0)
        o_spike = self.mul(i_spike, input_1)
        self.decoder(o_spike)

    
def test_mul_csg():
    """
    Test mul_csg with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config={
        'mode': 'bipolar',
        'timestep': 1024,
        'generator': 'sobol',
    }
    mul_config=codec_config

    # Generate random inputs based on mode
    input_0 = gen_rand_tensor(codec_config['mode'], shape=(1000,), bitwidth=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)
    input_1 = gen_rand_tensor(codec_config['mode'], shape=(1000,), bitwidth=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_mul_csg instance
    mul_inst = napl_mul_csg(codec_config, mul_config).to(device)
    mul_inst(input_0, input_1, timesteps=codec_config['timestep'])

    # calculate the reference output
    r_value = input_0 * input_1

    # report the error
    report_error(mul_inst.decoder.spike_value, r_value)

    mul_inst.reset()

    print('Test passed.')


if __name__ == '__main__':
    test_mul_csg()


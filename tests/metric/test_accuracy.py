import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import mul_csg
from napl.metric import accuracy, report_error


class napl_mul_csg(napl_base):
    def __init__(self, codec_config, mul_csg_config, acc_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.mul_csg = mul_csg(mul_csg_config)
        self.accuracy = accuracy(acc_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input_0)
        o_spike = self.mul_csg(i_spike, input_1)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def test_accuracy():
    """
    Test mul_csg with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    mul_csg_config=codec_config
    acc_config={
        'polarity': 'bipolar',
        'name': 'mul_csg_inst',
    }

    # Generate random inputs based on polarity
    input_0 = gen_rand_tensor(codec_config['polarity'], shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)
    input_1 = gen_rand_tensor(codec_config['polarity'], shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_mul_csg instance
    mul_csg_inst = napl_mul_csg(codec_config, mul_csg_config, acc_config).to(device)
    mul_csg_inst(input_0, input_1, timesteps=codec_config['timestep'])

    # calculate the reference output
    r_value = input_0 * input_1

    # report the error
    mul_csg_inst.accuracy.report_error(r_value, verbose=True)
    report_error(mul_csg_inst.decoder.spike_value, r_value)

    mul_csg_inst.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_accuracy()


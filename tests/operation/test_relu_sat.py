import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import relu_sat
from napl.metric import report_error


class napl_relu_sat(napl_base):
    def __init__(self, codec_config, relu_sat_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.relu_sat = relu_sat(relu_sat_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.relu_sat(i_spike)
        self.decoder(o_spike)

    
def test_relu_sat():
    """
    Test relu_sat with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    relu_sat_config={
        'width': 3,
    }
    
    # Generate random inputs based on polarity, ensure positive numbers
    input = gen_rand_tensor('bipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)
    # input = gen_arange_tensor('unipolar', width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_relu_sat instance
    relu_sat_inst = napl_relu_sat(codec_config, relu_sat_config).to(device)
    relu_sat_inst(input, timesteps=codec_config['timestep'])

    # calculate the reference output
    r_value = torch.nn.ReLU()(input)

    # report the error
    report_error(relu_sat_inst.decoder.spike_value, r_value)

    assert relu_sat_inst.relu_sat.timestep_cur == codec_config['timestep']
    relu_sat_inst.reset()

    print('Test passed.')


if __name__ == '__main__':
    test_relu_sat()


import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import sigmoid_hard
from napl.metric import report_error


class napl_sigmoid_hard(napl_base):
    def __init__(self, codec_config, sigmoid_hard_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.sigmoid_hard = sigmoid_hard(sigmoid_hard_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.sigmoid_hard(i_spike)
        self.decoder(o_spike)

    
def test_sigmoid_hard():
    """
    Test sigmoid_hard with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    sigmoid_hard_config={
        'polarity': 'bipolar',
    }
    
    # Generate random inputs based on polarity, ensure positive numbers
    input = gen_rand_tensor('bipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)
    # input = gen_arange_tensor('unipolar', width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_sigmoid_hard instance
    sigmoid_hard_inst = napl_sigmoid_hard(codec_config, sigmoid_hard_config).to(device)
    sigmoid_hard_inst(input, timesteps=codec_config['timestep'])

    # calculate the reference output
    r_value = torch.nn.Hardsigmoid()(input * 3)

    # report the error
    report_error(sigmoid_hard_inst.decoder.spike_value, r_value)

    assert sigmoid_hard_inst.sigmoid_hard.timestep_cur == codec_config['timestep']
    sigmoid_hard_inst.reset()

    print('Test passed.')


if __name__ == '__main__':
    test_sigmoid_hard()


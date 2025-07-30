import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import delay
from napl.metric import report_error


class napl_delay(napl_base):
    def __init__(self, codec_config, delay_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.delay = delay(delay_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.delay(i_spike)
        self.decoder(o_spike)

    
def test_delay():
    """
    Test delay with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config={
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    delay_config={
        'delay': 1
    }
    
    # Generate random inputs based on mode
    input = gen_rand_tensor(codec_config['mode'], shape=(10,), bitwidth=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_delay instance
    delay_inst = napl_delay(codec_config, delay_config).to(device)
    delay_inst(input, timesteps=codec_config['timestep'])

    # calculate the reference output
    r_value = input

    # report the error
    report_error(delay_inst.decoder.spike_value, r_value)

    delay_inst.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_delay()


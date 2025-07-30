import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import sqrt_traceiscb
from napl.metric import report_error


class napl_sqrt_traceiscb(napl_base):
    def __init__(self, codec_config, sqrt_traceiscb_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.sqrt_traceiscb = sqrt_traceiscb(sqrt_traceiscb_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.sqrt_traceiscb(i_spike)
        self.decoder(o_spike)

    
def test_sqrt_traceiscb():
    """
    Test sqrt_traceiscb with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config={
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    sqrt_traceiscb_config={
        'mode': 'bipolar',
    }
    
    # Generate random inputs based on mode, ensure positive numbers
    input = gen_rand_tensor('unipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_sqrt_traceiscb instance
    sqrt_traceiscb_inst = napl_sqrt_traceiscb(codec_config, sqrt_traceiscb_config).to(device)
    sqrt_traceiscb_inst(input, timesteps=codec_config['timestep'])

    # calculate the reference output
    r_value = torch.sqrt(input)

    # report the error
    report_error(sqrt_traceiscb_inst.decoder.spike_value, r_value)

    sqrt_traceiscb_inst.reset()

    print('Test passed.')


if __name__ == '__main__':
    test_sqrt_traceiscb()


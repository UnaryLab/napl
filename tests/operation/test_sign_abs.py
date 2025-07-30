import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import sign_abs
from napl.metric import report_error


class napl_sign_abs(napl_base):
    def __init__(self, codec_config, sign_abs_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder_sign = decoder(codec_config)
        self.decoder_abs = decoder(codec_config)
        self.sign_abs = sign_abs(sign_abs_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike_sign, o_spike_abs = self.sign_abs(i_spike)
        self.decoder_sign(o_spike_sign)
        self.decoder_abs(o_spike_abs)

    
def test_sign_abs():
    """
    Test sign_abs with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config={
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    sign_abs_config={
        'width': 3
    }
    
    # Generate random inputs based on mode
    input = gen_rand_tensor(codec_config['mode'], shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_sign_abs instance
    sign_abs_inst = napl_sign_abs(codec_config, sign_abs_config).to(device)
    sign_abs_inst(input, timesteps=codec_config['timestep'])

    # calculate the reference output
    # torch.sign is -1 for negative, and 1 for positive
    # napl.sign_abs is spike 0 means positive, i.e., bipolar -1
    # napl.sign_abs is spike 1 means negative, i.e., bipolar 1
    r_value_sign = torch.sign(input) * (-1)
    r_value_abs = torch.abs(input)

    # report the error
    report_error(sign_abs_inst.decoder_sign.spike_value, r_value_sign)
    report_error(sign_abs_inst.decoder_abs.spike_value, r_value_abs)

    sign_abs_inst.reset()

    print('Test passed.')


if __name__ == '__main__':
    test_sign_abs()


import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import add_any
from napl.metric import report_error


class napl_add_any(napl_base):
    def __init__(self, codec_config, add_any_config):
        super().__init__()
        # set up encoder, decoder, add_any, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.add_any = add_any(add_any_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.add_any(i_spike, dim=-1)
        self.decoder(o_spike)

    
def test_add_any():
    """
    Test add_any with a simple configuration.
    """
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config={
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    add_any_config={
        'mode': 'bipolar',
        'scale': 128,
        'width': 20,
    }

    # Generate random inputs based on mode
    input = gen_rand_tensor(codec_config['mode'], 
                          shape=(1000, add_any_config['scale']), 
                          width=math.log2(codec_config['timestep'])
                          ).type(global_config.ntype).to(device)

    # generate the napl_add_any instance
    add_any_inst = napl_add_any(codec_config, add_any_config).to(device)
    add_any_inst(input, timesteps=codec_config['timestep'])

    # calculate the reference output
    r_value = torch.sum(input, dim=-1) / add_any_config['scale']
    
    # report the error
    report_error(add_any_inst.decoder.spike_value, r_value)

    assert add_any_inst.add_any.timestep_cur == codec_config['timestep']
    add_any_inst.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_add_any()


import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps_class
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import add_any
from napl.metric import accuracy


class napl_add_any(napl_base):
    def __init__(self, codec_config, add_config, acc_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.adder = add_any(add_config)
        self.accuracy = accuracy(acc_config)


    @napl_sim_timesteps_class
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.adder(i_spike, dim=-1)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def test_add_any():
    """
    Test add_any with a simple configuration.
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    codec_config={
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    add_config={
        'mode': 'bipolar',
        'scale': 128,
        'bitwidth': 20,
    }
    acc_config={
        'mode': 'bipolar',
        'name': 'add_inst',
    }

    # Generate random inputs based on mode
    input = gen_rand_tensor(codec_config['mode'], 
                          shape=(1000, add_config['scale']), 
                          bitwidth=math.log2(codec_config['timestep'])
                          ).type(global_config.ntype).to(device)

    # generate the napl_add_any instance
    add_inst = napl_add_any(codec_config, add_config, acc_config).to(device)
    add_inst(input, timesteps=codec_config['timestep'])

    # calculate the reference output
    r_value = torch.sum(input, dim=-1) / add_config['scale']
    
    # report the error
    add_inst.accuracy.report_error(r_value)

    print("Add test passed.")


if __name__ == "__main__":
    test_add_any()


import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import sync_skewed
from napl.metric import report_error


class napl_sync_skewed(napl_base):
    def __init__(self, codec_config1, codec_config2, sync_skewed_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder0 = decoder(codec_config1)
        self.decoder1 = decoder(codec_config1)
        self.sync_skewed = sync_skewed(sync_skewed_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike0, o_spike1 = self.sync_skewed(i_spike0, i_spike1)
        self.decoder0(o_spike0)
        self.decoder1(o_spike1)

    
def test_sync_skewed():
    """
    Test sync_skewed with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config1={
        'mode': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'mode': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 3,
    }
    sync_skewed_config={
        'width' : 3,
    }

    # Generate random inputs based on mode
    input_0 = gen_rand_tensor(codec_config1['mode'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype).to(device)
    input_1 = gen_rand_tensor(codec_config2['mode'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype).to(device)
    input_mask = input_0 < input_1
    input_0_new = torch.where(input_mask, input_0, input_1)
    input_1_new = torch.where(~input_mask, input_0, input_1)
    # make sure divisor is not 0
    input_1_new = torch.where(input_1_new==0, 1, input_1_new)
    input_0 = input_0_new
    input_1 = input_1_new

    # generate the napl_sync_skewed instance
    sync_skewed_inst = napl_sync_skewed(codec_config1, codec_config2, sync_skewed_config).to(device)
    sync_skewed_inst(input_0, input_1, timesteps=codec_config1['timestep'])

    # calculate the reference output
    r_value0, r_value1 = input_0, input_1

    # report the error
    report_error(sync_skewed_inst.decoder0.spike_value, r_value0)
    report_error(sync_skewed_inst.decoder1.spike_value, r_value1)

    assert sync_skewed_inst.sync_skewed.timestep_cur == codec_config1['timestep']
    sync_skewed_inst.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_sync_skewed()


import torch, math

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import gt_rc
from napl.metric import report_error


class napl_gt_rc(napl_base):
    def __init__(self, codec_config1, codec_config2, codec_config3, gt_rc_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder = decoder(codec_config3)
        self.gt_rc = gt_rc(gt_rc_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.gt_rc(i_spike0, i_spike1)
        self.decoder(o_spike)

    
def test_gt_rc():
    """
    Test gt_rc with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config1={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 2,
    }
    codec_config3={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 2,
    }
    gt_rc_config=codec_config1

    # Generate random inputs based on polarity
    input_0 = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype).to(device)
    input_1 = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_gt_rc instance
    gt_rc_inst = napl_gt_rc(codec_config1, codec_config2, codec_config3, gt_rc_config).to(device)
    gt_rc_inst(input_0, input_1, timesteps=codec_config1['timestep'])

    # calculate the reference output
    r_value = (input_0 > input_1).type(global_config.ntype)

    # report the error
    _, idx = report_error(gt_rc_inst.decoder.spike_value, r_value)

    print(f'index with the largest error: {idx.item():7d}\tinput 0: {input_0[idx].item(): .5f}\tinput 1: {input_1[idx].item(): .5f}\tvalue gt_rc: {gt_rc_inst.decoder.spike_value[idx].item(): .5f}')
    
    assert gt_rc_inst.gt_rc.timestep_cur == codec_config1['timestep']
    gt_rc_inst.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_gt_rc()


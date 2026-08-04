import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.module import encoder, decoder
from napl.sim.operation import sync_skewed
from napl.sim.metric import accuracy


class napl_sync_skewed(napl_base):
    def __init__(self, codec_config1, codec_config2, sync_skewed_config):
        super().__init__()
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder0 = decoder(codec_config1)
        self.decoder1 = decoder(codec_config1)
        self.sync_skewed = sync_skewed(sync_skewed_config)
        self.accuracy0 = accuracy(codec_config1)
        self.accuracy1 = accuracy(codec_config1)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike0, o_spike1 = self.sync_skewed(i_spike0, i_spike1)
        self.decoder0(o_spike0)
        self.decoder1(o_spike1)
        self.accuracy0(o_spike0)
        self.accuracy1(o_spike1)

    
def _kernel_specific_checks():
    """
    Test sync_skewed with a simple configuration.
    """

    codec_config1={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 3,
    }
    sync_skewed_config={
        'width' : 3,
    }

    input_0_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)
    input_mask = input_0_cpu < input_1_cpu
    input_0_new = torch.where(input_mask, input_0_cpu, input_1_cpu)
    input_1_new = torch.where(~input_mask, input_0_cpu, input_1_cpu)
    input_1_new = torch.where(input_1_new==0, 1, input_1_new)
    input_0_cpu = input_0_new
    input_1_cpu = input_1_new

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        sync_skewed_inst = napl_sync_skewed(codec_config1, codec_config2, sync_skewed_config).to(device)
        with timer(device) as elapsed:
            sync_skewed_inst(input_0, input_1, timesteps=codec_config1['timestep'])

        sync_skewed_inst.accuracy0.analyze(input_0, verbose=True)
        sync_skewed_inst.accuracy1.analyze(input_1, verbose=True)
        assert sync_skewed_inst.sync_skewed.timestep_cur == codec_config1['timestep']
        sync_skewed_inst.reset()
        assert sync_skewed_inst.sync_skewed.timestep_cur == 0
        print(f'[{device}] time: {elapsed.seconds * 1000:.1f} ms')
    
    print('Test passed.')


def make_operation(_polarity, _timestep, _device):
    return sync_skewed({'width': 3})


def make_values(_polarity):
    first = torch.linspace(0.0, 1.0, 128)
    second = torch.linspace(1.0, 0.0, 128)
    return first, second


def make_performance_values(_polarity):
    first = torch.linspace(0.0, 1.0, 131072)
    second = torch.linspace(1.0, 0.0, 131072)
    return first, second


def analytic_reference(values, _polarity):
    return values[1]


def known_answer_case(_polarity):
    values = (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]))
    return values, values[1], 0.0


CONFIG = {
    'polarities': ['unipolar'],
    'tolerance_scale': 2.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [1, 3],
    'apply_operation': lambda operation, spikes: operation(*spikes)[1],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_sync_skewed():
    """Verify sync_skewed across the full unipolar range and input ordering."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_sync_skewed()

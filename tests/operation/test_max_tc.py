import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import decode, encode, max_tc
from napl.sim.metric import accuracy


class napl_max_tc(napl_base):
    def __init__(self, codec_config1, codec_config2, max_tc_config):
        super().__init__()
        self.encoder0 = encode(codec_config1)
        self.encoder1 = encode(codec_config2)
        self.decoder = decode(codec_config1)
        self.max_tc = max_tc(max_tc_config)
        self.accuracy = accuracy({'polarity': codec_config1['polarity']})


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.max_tc(i_spike0, i_spike1)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test max_tc with a simple configuration.
    """

    codec_config1={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'temporal',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'temporal',
        'dim': 2,
    }
    max_tc_config={'polarity': codec_config1['polarity']}

    input_0_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        max_tc_inst = napl_max_tc(codec_config1, codec_config2, max_tc_config).to(device)
        with timer(device) as elapsed:
            max_tc_inst(input_0, input_1, timesteps=codec_config1['timestep'])

        r_value = torch.max(input_0, input_1)
        error, _ = max_tc_inst.accuracy.analyze(r_value, verbose=True)
        rmse = error.pow(2).mean().sqrt()
        assert rmse <= CONFIG['tolerance_scale'] / math.sqrt(codec_config1['timestep']), rmse
        assert max_tc_inst.max_tc.timestep_cur == codec_config1['timestep']
        max_tc_inst.reset()
        assert max_tc_inst.max_tc.timestep_cur == 0
        print(f'[{device}] time: {elapsed.seconds * 1000:.1f} ms')
    
    print('Test passed.')


def make_operation(polarity, timestep, _device):
    return max_tc({
        'polarity': polarity,
    })


def make_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    left = torch.linspace(lo, hi, 128)
    return left, left.roll(31)


def make_performance_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    left = torch.linspace(lo, hi, 131072)
    return left, left.roll(31)


def analytic_reference(values, _polarity):
    return torch.maximum(values[0], values[1])


def known_answer_case(polarity):
    if polarity == 'unipolar':
        values = (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]))
    else:
        values = (torch.tensor([-1.0, 1.0]), torch.tensor([1.0, -1.0]))
    return values, torch.tensor([1.0, 1.0]), 0.0


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 3.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_generators': ['temporal', 'temporal'],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_max_tc():
    """Verify max_tc for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_max_tc()

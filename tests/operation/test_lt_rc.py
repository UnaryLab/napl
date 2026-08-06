import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import decode, encode, lt_rc
from napl.sim.metric import accuracy


class napl_lt_rc(napl_base):
    def __init__(self, codec_config1, codec_config2, codec_config3, lt_rc_config):
        super().__init__()
        self.encoder0 = encode(codec_config1)
        self.encoder1 = encode(codec_config2)
        self.decoder = decode(codec_config3)
        self.lt_rc = lt_rc(lt_rc_config)
        self.accuracy = accuracy({'polarity': codec_config3['polarity']})


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.lt_rc(i_spike0, i_spike1)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test lt_rc with a simple configuration.
    """
    torch.manual_seed(0)

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
    lt_rc_config={'polarity': codec_config1['polarity']}

    input_0_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        lt_rc_inst = napl_lt_rc(codec_config1, codec_config2, codec_config3, lt_rc_config).to(device)

        with timer(device) as elapsed:
            lt_rc_inst(input_0, input_1, timesteps=codec_config1['timestep'])

        r_value = (input_0 < input_1).type(global_config.ntype)
        error, result = lt_rc_inst.accuracy.analyze(r_value, verbose=True)
        rmse = error.pow(2).mean().sqrt().item()
        bound = 2.0 / math.sqrt(codec_config1['timestep'])
        assert rmse < bound, f'[{device}] rmse={rmse:.4f}, bound={bound:.4f}'

        print(f'[{device}] rmse={rmse:.4f}, bound={bound:.4f}, time={elapsed.seconds:.3f}s, '
              f'max-error index={result.max_absolute_index.item():7d}')
        assert lt_rc_inst.lt_rc.timestep_cur == codec_config1['timestep']
        lt_rc_inst.reset()
        assert lt_rc_inst.lt_rc.timestep_cur == 0
    
    print('Test passed.')


def make_operation(polarity, timestep, _device):
    return lt_rc({
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
    return (values[0] < values[1]).type(global_config.ntype)


def known_answer_case(polarity):
    if polarity == 'unipolar':
        values = (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]))
    else:
        values = (torch.tensor([-1.0, 1.0]), torch.tensor([1.0, -1.0]))
    return values, torch.tensor([1.0, 0.0]), 2.0 / math.sqrt(256)


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 2.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'output_polarity': 'unipolar',
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_lt_rc():
    """Verify lt_rc for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_lt_rc()

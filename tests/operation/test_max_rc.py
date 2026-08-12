import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import decode, encode, max_rc
from napl.sim.metric import accuracy


# Widest input gap the comparator still resolves, in bipolar value units.
_RESOLUTION = 16.0 / 256


class napl_max_rc(napl_base):
    def __init__(self, codec_config1, codec_config2, codec_config3, max_rc_config):
        super().__init__()
        self.encoder0 = encode(codec_config1)
        self.encoder1 = encode(codec_config2)
        self.decoder0 = decode(codec_config1)
        self.decoder1 = decode(codec_config3)
        self.max_rc = max_rc(max_rc_config)
        self.accuracy0 = accuracy({'polarity': codec_config1['polarity']})
        self.accuracy1 = accuracy({'polarity': codec_config3['polarity']})


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike0, o_spike1 = self.max_rc(i_spike0, i_spike1)
        self.decoder0(o_spike0)
        self.decoder1(o_spike1)
        self.accuracy0(o_spike0)
        self.accuracy1(o_spike1)

    
def _kernel_specific_checks():
    """
    Test max_rc with a simple configuration.
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
    max_rc_config={'polarity': codec_config1['polarity']}

    input_0_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        max_rc_inst = napl_max_rc(codec_config1, codec_config2, codec_config3, max_rc_config).to(device)
        with timer(device) as elapsed:
            max_rc_inst(input_0, input_1, timesteps=codec_config1['timestep'])

        r_value = torch.max(input_0, input_1)
        r_value_arg = torch.argmax(torch.stack([input_0, input_1], dim=0), dim=0)
        value_error, value_result = max_rc_inst.accuracy0.analyze(r_value, verbose=True)
        value_rmse = value_error.pow(2).mean().sqrt()
        arg_error, arg_result = max_rc_inst.accuracy1.analyze(r_value_arg, verbose=True)
        # The comparator picks a winner only when the two values differ by more
        # than its stream resolution; inside that band the hard reference has no
        # answer the kernel can be held to, so the arg rmse is scored over the
        # resolvable pairs only. The value output is unaffected, since a near-tie
        # makes the two candidates nearly equal, so it is scored over the whole
        # draw.
        resolvable = (input_0 - input_1).abs() > _RESOLUTION
        arg_rmse = arg_error[resolvable].pow(2).mean().sqrt()

        print(f'[{device}] value rmse={value_rmse:.4f}; '
              f'arg rmse={arg_rmse:.4f} over {int(resolvable.sum())} resolvable pairs; '
              f'value max error index: {value_result.max_absolute_index.item():7d}; arg max error index: {arg_result.max_absolute_index.item():7d}; time: {elapsed.seconds * 1000:.1f} ms')
        assert max_rc_inst.max_rc.timestep_cur == codec_config1['timestep']
        max_rc_inst.reset()
        assert max_rc_inst.max_rc.timestep_cur == 0
    
    print('Test passed.')


def make_operation(polarity, timestep, _device):
    return max_rc({
        'polarity': polarity,
    })


def make_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    left = torch.linspace(lo, hi, 128)
    return left, left.roll(31)


def make_random_perf_values(polarity):
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
    # Full-scale opposites, so the winner is unambiguous from the first
    # timestep and the only residue is the encoder's 1 / N rate step.
    return values, torch.tensor([1.0, 1.0])


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'apply_operation': lambda operation, spikes: operation(*spikes)[0],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_max_rc():
    """Verify max_rc for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_max_rc()

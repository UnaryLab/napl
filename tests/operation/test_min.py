import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import decode, encode, min
from napl.sim.metric import accuracy


# Widest input gap the comparator still resolves, in bipolar value units.
_RESOLUTION = 16.0 / 256


class napl_min(napl_base):
    def __init__(self, codec_config1, codec_config2, codec_config3, min_config):
        super().__init__()
        self.encoder0 = encode(codec_config1)
        self.encoder1 = encode(codec_config2)
        self.decoder0 = decode(codec_config1)
        self.decoder1 = decode(codec_config3)
        self.min = min(min_config)
        self.accuracy0 = accuracy({'polarity': codec_config1['polarity']})
        self.accuracy1 = accuracy({'polarity': codec_config3['polarity']})


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike0, o_spike1 = self.min(i_spike0, i_spike1)
        self.decoder0(o_spike0)
        self.decoder1(o_spike1)
        self.accuracy0(o_spike0)
        self.accuracy1(o_spike1)

    
def _kernel_specific_checks():
    """
    Test min with a simple configuration.
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
    min_config={'polarity': codec_config1['polarity']}

    input_0_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        min_inst = napl_min(codec_config1, codec_config2, codec_config3, min_config).to(device)
        with timer(device) as elapsed:
            min_inst(input_0, input_1, timesteps=codec_config1['timestep'])

        r_value = torch.min(input_0, input_1)
        r_value_arg = torch.argmin(torch.stack([input_0, input_1], dim=0), dim=0)
        value_error, value_result = min_inst.accuracy0.analyze(r_value, verbose=True)
        value_rmse = value_error.pow(2).mean().sqrt()
        arg_error, arg_result = min_inst.accuracy1.analyze(r_value_arg, verbose=True)
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
        assert min_inst.min.timestep_cur == codec_config1['timestep']
        min_inst.reset()
        assert min_inst.min.timestep_cur == 0
    
    print('Test passed.')


def make_operation(polarity, timestep, _device):
    return min({
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
    return torch.minimum(values[0], values[1])


def known_answer_case(polarity):
    if polarity == 'unipolar':
        values = (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]))
        expected = torch.tensor([0.0, 0.0])
    else:
        values = (torch.tensor([-1.0, 1.0]), torch.tensor([1.0, -1.0]))
        expected = torch.tensor([-1.0, -1.0])
    # Full-scale opposites, so the winner is unambiguous from the first
    # timestep and the only residue is the encoder's 1 / N rate step.
    return values, expected


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


def test_min():
    """Verify min for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


def test_min_rejects_zero_dim_operands():
    """Verify a pair of 0-dim operands raises a message naming the operation and the rank contract."""
    for device in devices():
        scalar = torch.ones((), dtype=global_config.stype, device=device)
        operation = min({'polarity': 'bipolar'}).to(device)
        try:
            operation(scalar, scalar)
        except AssertionError as error:
            expected = (
                'Invalid min operand rank: <0>; legal values: at least 1, since the '
                'selection state is held at rank 1.'
            )
            assert str(error) == expected, f'{str(error)!r} != {expected!r}'
        else:
            raise AssertionError(f'{device}: min accepted a pair of 0-dim operands')

        # Rank-1 operands stay accepted, so the rejection above is specific to the 0-dim pair.
        rank_1 = torch.ones(2, dtype=global_config.stype, device=device)
        output, index = min({'polarity': 'bipolar'}).to(device)(rank_1, rank_1)
        assert output.shape == rank_1.shape and index.shape == rank_1.shape

        print(f'[{device}] 0-dim operand pair rejected; rank-1 pair accepted')

    print('Test passed.')


def test_min_rejects_a_shape_change_without_reset():
    """Verify a mid-run input shape change raises and reset() readmits the new shape."""
    for device in devices():
        wide = torch.ones(4, dtype=global_config.stype, device=device)
        narrow = torch.ones(3, dtype=global_config.stype, device=device)
        operation = min({'polarity': 'bipolar'}).to(device)
        operation(wide, wide)
        try:
            operation(narrow, narrow)
        except RuntimeError as error:
            # The rejection is torch's broadcast error raised inside the synchronizer, so the
            # message is matched here rather than the RuntimeError type alone.
            expected = (
                'The size of tensor a (4) must match the size of tensor b (3) at '
                'non-singleton dimension 0'
            )
            assert expected in str(error), f'{str(error)!r} does not contain {expected!r}'
        else:
            raise AssertionError(f'{device}: min accepted a shape change without reset()')

        operation.reset()
        output, index = operation(narrow, narrow)
        assert output.shape == narrow.shape and index.shape == narrow.shape

        print(f'[{device}] mid-run shape change rejected; accepted after reset()')

    print('Test passed.')


def test_min_keeps_stale_state_on_a_broadcastable_shape_change():
    """Verify a mid-run change to a broadcastable shape is accepted and keeps the stale state."""
    for device in devices():
        wide_0 = torch.tensor([0.0, 1.0, 0.0], dtype=global_config.stype, device=device)
        wide_1 = torch.ones(3, dtype=global_config.stype, device=device)
        narrow_0 = torch.zeros(1, dtype=global_config.stype, device=device)
        narrow_1 = torch.ones(1, dtype=global_config.stype, device=device)
        operation = min({'polarity': 'bipolar'}).to(device)

        # The warm-up drives the selection state to a per-element pattern, so a state that the
        # shrunk call drops, re-zeroes, or refills with the reset value shows up below.
        operation(wide_0, wide_1)
        assert operation.index.tolist() == [1, 0, 1], f'[{device}] warm-up left the state at {operation.index.tolist()}'

        output, index = operation(narrow_0, narrow_1)

        # A shape-(1,) operand broadcasts into the shape-(3,) synchronizer counter and selection
        # state, so the call is accepted at the state shape and the output carries the
        # pre-update state.
        assert output.shape == wide_1.shape and index.shape == wide_1.shape
        assert output.tolist() == [0, 1, 0], f'[{device}] stale selection state lost; output {output.tolist()}'
        assert operation.index.shape == wide_1.shape

        print(f'[{device}] broadcastable shape change accepted; stale state kept as [0, 1, 0] at shape {tuple(output.shape)}')

    print('Test passed.')


if __name__ == '__main__':
    test_min()
    test_min_rejects_zero_dim_operands()
    test_min_rejects_a_shape_change_without_reset()
    test_min_keeps_stale_state_on_a_broadcastable_shape_change()

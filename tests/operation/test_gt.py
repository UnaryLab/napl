import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import decode, encode, gt
from napl.sim.metric import accuracy


# Widest input gap the comparator still resolves, in bipolar value units.
_RESOLUTION = 16.0 / 256


class napl_gt(napl_base):
    def __init__(self, codec_config1, codec_config2, codec_config3, gt_config):
        super().__init__()
        self.encoder0 = encode(codec_config1)
        self.encoder1 = encode(codec_config2)
        self.decoder = decode(codec_config3)
        self.accuracy = accuracy({'polarity': codec_config3['polarity']})
        self.gt = gt(gt_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.gt(i_spike0, i_spike1)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test gt with a simple configuration.
    """

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
    gt_config={'polarity': codec_config1['polarity']}

    input_0_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        gt_inst = napl_gt(codec_config1, codec_config2, codec_config3, gt_config).to(device)

        with timer(device) as elapsed:
            gt_inst(input_0, input_1, timesteps=codec_config1['timestep'])

        r_value = (input_0 > input_1).type(global_config.ntype)
        error, result = gt_inst.accuracy.analyze(r_value, verbose=True)
        # The comparator resolves a pair only when the two values differ by more
        # than its stream resolution; inside that band the hard reference has no
        # answer the kernel can be held to, so the printed rmse is scored over
        # the resolvable pairs only.
        resolvable = (input_0 - input_1).abs() > _RESOLUTION
        rmse = error[resolvable].pow(2).mean().sqrt().item()

        print(f'[{device}] rmse={rmse:.4f}, time={elapsed.seconds:.3f}s, '
              f'max-error index={result.max_absolute_index.item():7d}')
        assert gt_inst.gt.timestep_cur == codec_config1['timestep']
        gt_inst.reset()
        assert gt_inst.gt.timestep_cur == 0
    
    _first_timestep_shape_checks()
    print('Test passed.')


def _first_timestep_shape_checks():
    """
    Test that timestep 0 carries the input shape and a run stacks.
    """
    codec_config = {
        'polarity': 'bipolar',
        'timestep': 64,
        'generator': 'sobol',
    }

    input_0_cpu = gen_rand_tensor('bipolar', shape=(3, 4), width=math.log2(codec_config['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor('bipolar', shape=(3, 4), width=math.log2(codec_config['timestep'])).type(global_config.ntype)

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        encoder_0 = encode({**codec_config, 'dim': 1}).to(device)
        encoder_1 = encode({**codec_config, 'dim': 2}).to(device)
        operation = gt({'polarity': codec_config['polarity']}).to(device)

        outputs = [operation(encoder_0(input_0), encoder_1(input_1))
                   for _ in range(codec_config['timestep'])]

        assert outputs[0].shape == input_0.shape
        assert torch.equal(outputs[0], torch.full_like(outputs[0], 1))
        assert torch.stack(outputs).shape == (codec_config['timestep'], *input_0.shape)


def make_operation(polarity, timestep, _device):
    return gt({
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
    return (values[0] > values[1]).type(global_config.ntype)


def known_answer_case(polarity):
    if polarity == 'unipolar':
        values = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    else:
        values = (torch.tensor([1.0, -1.0]), torch.tensor([-1.0, 1.0]))
    return values, torch.tensor([1.0, 0.0])


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'output_polarity': 'unipolar',
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_gt():
    """Verify gt for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


def test_gt_rejects_zero_dim_operands():
    """Verify a pair of 0-dim operands raises a message naming the operation and the rank contract."""
    for device in devices():
        scalar = torch.ones((), dtype=global_config.stype, device=device)
        operation = gt({'polarity': 'bipolar'}).to(device)
        try:
            operation(scalar, scalar)
        except AssertionError as error:
            expected = (
                'Invalid gt operand rank: <0>; legal values: at least 1, since the '
                'decision state is held at rank 1.'
            )
            assert str(error) == expected, f'{str(error)!r} != {expected!r}'
        else:
            raise AssertionError(f'{device}: gt accepted a pair of 0-dim operands')

        # Rank-1 operands stay accepted, so the rejection above is specific to the 0-dim pair.
        rank_1 = torch.ones(2, dtype=global_config.stype, device=device)
        output = gt({'polarity': 'bipolar'}).to(device)(rank_1, rank_1)
        assert output.shape == rank_1.shape

        print(f'[{device}] 0-dim operand pair rejected; rank-1 pair accepted')

    print('Test passed.')


def test_gt_rejects_a_shape_change_without_reset():
    """Verify a mid-run input shape change raises and reset() readmits the new shape."""
    for device in devices():
        wide = torch.ones(4, dtype=global_config.stype, device=device)
        narrow = torch.ones(3, dtype=global_config.stype, device=device)
        operation = gt({'polarity': 'bipolar'}).to(device)
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
            raise AssertionError(f'{device}: gt accepted a shape change without reset()')

        operation.reset()
        output = operation(narrow, narrow)
        assert output.shape == narrow.shape

        print(f'[{device}] mid-run shape change rejected; accepted after reset()')

    print('Test passed.')


def test_gt_keeps_stale_state_on_a_broadcastable_shape_change():
    """Verify a mid-run change to a broadcastable shape is accepted and keeps the stale state."""
    for device in devices():
        wide_0 = torch.tensor([0.0, 1.0, 0.0], dtype=global_config.stype, device=device)
        wide_1 = torch.ones(3, dtype=global_config.stype, device=device)
        narrow_0 = torch.zeros(1, dtype=global_config.stype, device=device)
        narrow_1 = torch.ones(1, dtype=global_config.stype, device=device)
        operation = gt({'polarity': 'bipolar'}).to(device)

        # The warm-up drives the decision state to a per-element pattern, so a state that the
        # shrunk call drops, re-zeroes, or refills with the reset value shows up below.
        operation(wide_0, wide_1)
        assert operation.decision.tolist() == [0, 1, 0], f'[{device}] warm-up left the state at {operation.decision.tolist()}'

        output = operation(narrow_0, narrow_1)

        # A shape-(1,) operand broadcasts into the shape-(3,) synchronizer counter and decision
        # state, so the call is accepted at the state shape and the output carries the
        # pre-update state.
        assert output.shape == wide_1.shape
        assert output.tolist() == [0, 1, 0], f'[{device}] stale decision state lost; output {output.tolist()}'
        assert operation.decision.shape == wide_1.shape

        print(f'[{device}] broadcastable shape change accepted; stale state kept as [0, 1, 0] at shape {tuple(output.shape)}')

    print('Test passed.')


if __name__ == '__main__':
    test_gt()
    test_gt_rejects_zero_dim_operands()
    test_gt_rejects_a_shape_change_without_reset()
    test_gt_keeps_stale_state_on_a_broadcastable_shape_change()

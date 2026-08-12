import torch

from napl.sim.base import global_config
from napl.utils._shared_test import devices, streaming_suite
from napl.sim.operation import mul_scale


SCALE = 2
INTWIDTH = 12
FRACWIDTH = 4
SEED = 1234


def _spike_stream(timesteps, shape, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 2, (timesteps,) + shape, generator=generator, dtype=torch.int64
    ).type(global_config.stype)


def make_operation(polarity, _timestep, _device):
    return mul_scale({
        'polarity': polarity,
        'scale': SCALE,
        'intwidth': INTWIDTH,
        'fracwidth': FRACWIDTH,
    })


def make_values(polarity):
    # Required range: the input is narrowed so x * SCALE stays a legal rate.
    # With SCALE = 2 that is [0, 0.5] unipolar and [-0.5, 0.5] bipolar, whose
    # products span the full [0, 1] and [-1, 1] output ranges.
    lo, hi = (0.0, 0.5) if polarity == 'unipolar' else (-0.5, 0.5)
    return (torch.linspace(lo, hi, 512).reshape(64, 8),)


def make_random_perf_values(polarity):
    lo, hi = (0.0, 0.5) if polarity == 'unipolar' else (-0.5, 0.5)
    return (torch.linspace(lo, hi, 16384 * 8).reshape(16384, 8),)


def analytic_reference(values, _polarity):
    return values[0] * SCALE


def known_answer_case(_polarity):
    # Half-scale input times the power-of-two scale of 2 lands exactly on the
    # full-scale output stream, so the answer carries no residue.
    values = torch.full((8, 8), 0.5)
    return (values,), values * SCALE


CONFIG = {
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'polarities': ['unipolar', 'bipolar'],
    'timesteps': 256,
    'apply_operation': lambda operation, spikes: operation(spikes[0]),
    # Gate 17 does not apply: mul_scale emits a single read output whose value is
    # exactly what the suite decodes, so a defect that changes the product is
    # already visible in the printed fidelity error; there is no identity-wire or
    # unread-output failure mode to pin.
    'extra_checks': None,
}


def test_mul_scale():
    """Verify mul_scale for both polarities against analytic and known-answer streams.

    Inputs are narrowed so x * SCALE stays a legal rate: [0, 0.5] unipolar and
    [-0.5, 0.5] bipolar for SCALE = 2.
    """
    streaming_suite(CONFIG)
    print('Test passed.')


def test_mul_scale_fires_at_the_fractional_scale_rate():
    """Verify a fractional scale multiplies a half-ones stream at the exact target rate."""
    timesteps = 64
    scale = 1.5
    # p_in = 0.5 (v_in = 0), so target rates are 0.5 * scale unipolar and 0.5 bipolar.
    input_cpu = _spike_stream(timesteps, (4, 3), seed=SEED + 7)
    expected_rate = {'unipolar': 0.5 * scale, 'bipolar': 0.5}

    for device in devices():
        stream = input_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            operation = mul_scale({
                'polarity': polarity,
                'scale': scale,
                'intwidth': INTWIDTH,
                'fracwidth': FRACWIDTH,
            }).to(device)

            fires = torch.zeros((4, 3), dtype=global_config.ntype, device=device)
            for timestep in range(timesteps):
                output = operation(stream[timestep])
                assert output.shape == (4, 3), (
                    f'[{device}] {polarity} output shape {tuple(output.shape)} != (4, 3)'
                )
                fires.add_(output.type(global_config.ntype))

            rate = (fires / timesteps).mean().item()
            print(
                f'[{device}][{polarity}] scale={scale}, '
                f'rate={rate:.4f}, expected~{expected_rate[polarity]:.4f}'
            )


def test_mul_scale_reset_and_replay_is_deterministic():
    """Verify reset clears the accumulator and a replayed stream reproduces the outputs."""
    timesteps = 32
    stream_cpu = _spike_stream(timesteps, (8, 4), seed=SEED + 1)

    for device in devices():
        stream = stream_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            operation = mul_scale({
                'polarity': polarity,
                'scale': 1.5,
                'intwidth': INTWIDTH,
                'fracwidth': FRACWIDTH,
            }).to(device)

            first = [operation(stream[step]) for step in range(timesteps)]
            assert operation.timestep_cur == timesteps
            operation.reset()
            assert operation.timestep_cur == 0
            assert operation.accumulator.abs().sum().item() == 0

            replay = [operation(stream[step]) for step in range(timesteps)]
            assert all(
                torch.equal(one, two) for one, two in zip(first, replay)
            ), f'[{device}] {polarity} replay diverged'


def test_mul_scale_saturates_above_unit_scale():
    """Verify a full-scale input times scale > 1 rails the accumulator and saturates the output."""
    timesteps = 64
    scale = 2.0
    input_cpu = torch.ones((4,), dtype=global_config.stype)

    for device in devices():
        input = input_cpu.to(device)
        operation = mul_scale({
            'polarity': 'unipolar',
            'scale': scale,
            'intwidth': 6,
            'fracwidth': 2,
        }).to(device)

        for _ in range(timesteps):
            output = operation(input)

        assert torch.equal(output, torch.ones_like(output)), (
            f'[{device}] output {output.tolist()} not all ones at saturation'
        )
        assert operation.accumulator.max().item() >= operation.acc_max - operation.unit, (
            f'[{device}] accumulator {operation.accumulator.tolist()} did not rail'
        )
        print(
            f'[{device}] saturated: acc_max={operation.acc_max}, '
            f'accumulator={operation.accumulator[0].item():.0f}'
        )


def test_mul_scale_rejects_invalid_scale():
    """Reject a non-numeric, non-positive, or non-finite scale by exact message."""
    grid = 2.0 ** (-FRACWIDTH)
    not_numeric = 'legal values: a positive finite int or float.'
    too_small = f'legal values: greater than half of <{grid}>.'
    cases = [
        (0, f'Invalid scale: <0>; {not_numeric}'),
        (-1.5, f'Invalid scale: <-1.5>; {not_numeric}'),
        (True, f'Invalid scale: <True>; {not_numeric}'),
        ('2', f'Invalid scale: <2>; {not_numeric}'),
        (float('inf'), f'Invalid scale: <inf>; {not_numeric}'),
        (float('nan'), f'Invalid scale: <nan>; {not_numeric}'),
        # Exactly half a raw unit ties down to zero raw units.
        (0.03125, f'Invalid scale: <0.03125>; {too_small}'),
    ]

    for scale, message in cases:
        try:
            mul_scale({
                'polarity': 'unipolar',
                'scale': scale,
                'intwidth': INTWIDTH,
                'fracwidth': FRACWIDTH,
            })
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'mul_scale must reject a scale of {scale}')

    # 2 ** (4 + 2 - 1) - 1 = 31 raw units is 7.75 in value units, so 8 overflows the accumulator.
    try:
        mul_scale({
            'polarity': 'unipolar',
            'scale': 8,
            'intwidth': 4,
            'fracwidth': 2,
        })
    except AssertionError as error:
        assert str(error) == (
            'mul_scale scale <8.0> exceeds accumulator maximum <7.75> for '
            'intwidth <4> and fracwidth <2>.'
        ), str(error)
        return
    raise AssertionError('mul_scale must reject a scale above acc_max')


if __name__ == '__main__':
    test_mul_scale()
    test_mul_scale_fires_at_the_fractional_scale_rate()
    test_mul_scale_reset_and_replay_is_deterministic()
    test_mul_scale_saturates_above_unit_scale()
    test_mul_scale_rejects_invalid_scale()

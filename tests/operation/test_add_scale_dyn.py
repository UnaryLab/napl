import torch

from napl.sim.base import global_config
from napl.utils._shared_test import devices, streaming_suite
from napl.sim.operation import add_scale, add_scale_dyn


ENTRY = 8
SCALE = ENTRY
SCALE_MAX = 16
INTWIDTH = 20
FRACWIDTH = 4
SEED = 1234


def _spike_stream(timesteps, shape, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 2, (timesteps,) + shape, generator=generator, dtype=torch.int64
    ).type(global_config.stype)


def make_operation(polarity, _timestep, _device):
    return add_scale_dyn({
        'polarity': polarity,
        'scale_max': SCALE_MAX,
        'intwidth': INTWIDTH,
        'fracwidth': FRACWIDTH,
    })


def make_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(lo, hi, 512).reshape(64, ENTRY),)


def make_random_perf_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(lo, hi, 16384 * ENTRY).reshape(16384, ENTRY),)


def analytic_reference(values, _polarity):
    return values[0].mean(dim=-1)


def known_answer_case(_polarity):
    values = torch.ones((ENTRY, ENTRY))
    # Every input is the full-scale stream, so the scaled adder selects among
    # identical streams and reproduces them spike for spike: the answer is
    # exact on both devices.
    return (values,), values.mean(dim=-1)


CONFIG = {
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'polarities': ['unipolar', 'bipolar'],
    'timesteps': 256,
    'apply_operation': lambda operation, spikes: operation(
        spikes[0], SCALE, entry=ENTRY, dim=-1
    ),
}


def test_add_scale_dyn():
    """Verify add_scale_dyn at a held scale against analytic and known-answer streams."""
    streaming_suite(CONFIG)


def test_add_scale_dyn_matches_add_scale_at_constant_scale():
    """Verify add_scale_dyn reproduces add_scale bit-exactly when the scale never changes."""
    timesteps = 128
    stream_cpu = _spike_stream(timesteps, (16, ENTRY))

    compared = 0
    for device in devices():
        stream = stream_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            # scale below, at, and above the adder fan-in, plus two off-integer scales.
            for scale in (4, ENTRY, 16, 2.5, 6.25):
                # fracwidth 0 keeps the accumulator on integers and cannot represent an
                # off-integer scale, so the fractional scales run only at fracwidth 4.
                for fracwidth in ((0, FRACWIDTH) if float(scale).is_integer() else (FRACWIDTH,)):
                    reference = add_scale({
                        'polarity': polarity,
                        'scale': scale,
                        'intwidth': INTWIDTH,
                        'fracwidth': fracwidth,
                    }).to(device)
                    candidate = add_scale_dyn({
                        'polarity': polarity,
                        'scale_max': SCALE_MAX,
                        'intwidth': INTWIDTH,
                        'fracwidth': fracwidth,
                    }).to(device)
                    for timestep in range(timesteps):
                        expected = reference(stream[timestep], dim=-1)
                        observed = candidate(stream[timestep], scale, dim=-1)
                        assert observed.shape == (16,), (
                            f'[{device}] {polarity} scale={scale} output shape '
                            f'{tuple(observed.shape)} != (16,)'
                        )
                        assert torch.equal(observed, expected), (
                            f'[{device}] {polarity} scale={scale} '
                            f'fracwidth={fracwidth} diverged at timestep {timestep}'
                        )
                    assert candidate.accumulator.shape == (16,)
                    assert torch.equal(candidate.accumulator, reference.accumulator)
                    assert candidate.scale == reference.scale
                    compared += 1
        print(f'[{device}] add_scale_dyn matched add_scale on 16 polarity/scale/fracwidth cases')

    assert compared == 16 * len(devices())


def test_add_scale_dyn_fires_at_the_fractional_scale_rate():
    """Verify a fractional per-call scale reduces an all-ones stream at the exact target rate."""
    timesteps = 60
    scale = 2.5
    entry = 2
    input_cpu = torch.ones((4, entry), dtype=global_config.stype)
    # An all-ones stream carries p_i = 1 (v_i = 1) on each of the two entries, so the target
    # rates are sum(p_i) / scale unipolar and (1 + sum(v_i) / scale) / 2 bipolar.
    expected_rate = {'unipolar': entry / scale, 'bipolar': (1.0 + entry / scale) / 2}

    for device in devices():
        input = input_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            operation = add_scale_dyn({
                'polarity': polarity,
                'scale_max': SCALE_MAX,
                'intwidth': INTWIDTH,
                'fracwidth': FRACWIDTH,
            }).to(device)

            fires = torch.zeros((4,), dtype=global_config.ntype, device=device)
            for _ in range(timesteps):
                output = operation(input, scale, dim=-1)
                assert output.shape == (4,), (
                    f'[{device}] {polarity} output shape {tuple(output.shape)} != (4,)'
                )
                fires.add_(output.type(global_config.ntype))

            rate = fires / timesteps
            expected = torch.full_like(rate, expected_rate[polarity])
            torch.testing.assert_close(rate, expected, atol=1e-6, rtol=0)
            print(
                f'[{device}][{polarity}] scale={scale}, entry={entry}, '
                f'rate={rate[0].item():.6f}, '
                f'expected={expected_rate[polarity]:.6f}'
            )


def test_add_scale_dyn_reports_the_effective_call_scale():
    """Verify the reported scale is the quantized request and resets to None."""
    input = torch.ones((2, ENTRY), dtype=global_config.stype)
    operation = add_scale_dyn({
        'polarity': 'unipolar',
        'scale_max': SCALE_MAX,
        'intwidth': INTWIDTH,
        'fracwidth': FRACWIDTH,
    })

    assert operation.scale is None
    # 2.03 sits between grid points and rounds down onto 32 raw units, which is 2.0.
    for requested, effective in ((4, 4.0), (2.5, 2.5), (2.03, 2.0)):
        operation(input, requested, dim=-1)
        assert operation.scale == effective, (
            f'scale {requested} reported as {operation.scale}, expected {effective}'
        )

    operation.reset()
    assert operation.scale is None


def test_add_scale_dyn_conserves_mass_across_a_scale_change():
    """Verify fired mass plus the accumulator equals the input mass across a scale change."""
    timesteps = 64
    early_scale, late_scale = float(ENTRY), 2.5
    unit = 2 ** FRACWIDTH
    stream_cpu = _spike_stream(timesteps, (4, ENTRY), seed=SEED + 1)

    for device in devices():
        stream = stream_cpu.to(device)
        operation = add_scale_dyn({
            'polarity': 'unipolar',
            'scale_max': SCALE_MAX,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }).to(device)

        fired_mass = torch.zeros(4, dtype=global_config.ntype, device=device)
        for timestep in range(timesteps):
            scale = early_scale if timestep < timesteps // 2 else late_scale
            output = operation(stream[timestep], scale, dim=-1)
            fired_mass.add_(output.type(global_config.ntype), alpha=round(scale * unit))

        total_input = stream.type(global_config.ntype).sum(dim=(0, -1)) * unit
        residual = total_input - fired_mass
        # The accumulator drains by at most the pre-change scale, so it ends the
        # first half below ENTRY and then gains at most (ENTRY - late_scale) per
        # remaining timestep, all in raw units.
        bound = (ENTRY * unit - 1) + (timesteps // 2) * round((ENTRY - late_scale) * unit)
        max_residual = residual.max().item()

        assert torch.equal(residual, operation.accumulator), (
            f'[{device}] residual {residual.tolist()} != accumulator '
            f'{operation.accumulator.tolist()}'
        )
        assert residual.min().item() >= 0, f'[{device}] negative residual'
        assert max_residual <= bound, (
            f'[{device}] residual {max_residual} exceeds bound {bound}'
        )
        assert max_residual < operation.acc_max, f'[{device}] accumulator clamped'
        print(
            f'[{device}] input={total_input.sum().item():.0f}, '
            f'fired={fired_mass.sum().item():.0f}, '
            f'residual max={max_residual:.0f}, bound={bound} raw units'
        )


def test_add_scale_dyn_conserves_mass_across_a_scale_change_bipolar():
    """Verify the bipolar offset tracks the current call scale across a scale change."""
    timesteps = 64
    early_scale, late_scale = float(ENTRY), 2.5
    unit = 2 ** FRACWIDTH
    stream_cpu = _spike_stream(timesteps, (4, ENTRY), seed=SEED + 1)

    for device in devices():
        stream = stream_cpu.to(device)
        operation = add_scale_dyn({
            'polarity': 'bipolar',
            'scale_max': SCALE_MAX,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }).to(device)

        fired_mass = torch.zeros(4, dtype=global_config.ntype, device=device)
        expected_total = torch.zeros(4, dtype=global_config.ntype, device=device)
        for timestep in range(timesteps):
            scale = early_scale if timestep < timesteps // 2 else late_scale
            scale_raw = round(scale * unit)
            expected_total.add_(
                stream[timestep].type(global_config.ntype).sum(dim=-1) * unit
                - (ENTRY * unit - scale_raw) / 2
            )
            output = operation(stream[timestep], scale, dim=-1)
            fired_mass.add_(output.type(global_config.ntype), alpha=scale_raw)

        residual = expected_total - fired_mass
        assert torch.equal(residual, operation.accumulator), (
            f'[{device}] residual {residual.tolist()} != accumulator '
            f'{operation.accumulator.tolist()}'
        )
        assert residual.abs().max().item() < operation.acc_max, (
            f'[{device}] accumulator clamped'
        )
        print(
            f'[{device}] bipolar expected={expected_total.sum().item():.1f}, '
            f'fired={fired_mass.sum().item():.0f}, '
            f'residual max={residual.max().item():.1f} raw units'
        )


def test_add_scale_dyn_saturates_at_the_accumulator_rail():
    """Verify a narrow accumulator clamps at acc_max before the carry is subtracted."""
    timesteps = 40
    scale = 0.5
    input_cpu = torch.ones((4, ENTRY), dtype=global_config.stype)

    for device in devices():
        input = input_cpu.to(device)
        operation = add_scale_dyn({
            'polarity': 'unipolar',
            'scale_max': SCALE_MAX,
            'intwidth': 6,
            'fracwidth': 2,
        }).to(device)

        for _ in range(timesteps):
            output = operation(input, scale, dim=-1)

        # Inflow of ENTRY per timestep outruns the drain of scale, so the clamp
        # holds the accumulator at acc_max and the carry leaves acc_max - scale_raw.
        railed = operation.acc_max - round(scale * 2**operation.fracwidth)
        assert torch.equal(
            operation.accumulator,
            torch.full_like(operation.accumulator, railed),
        ), (
            f'[{device}] accumulator {operation.accumulator.tolist()} != '
            f'acc_max - scale_raw {railed}'
        )
        assert torch.equal(output, torch.ones_like(output)), (
            f'[{device}] output {output.tolist()} not all ones at saturation'
        )
        print(
            f'[{device}] saturated: acc_max={operation.acc_max}, '
            f'accumulator={operation.accumulator[0].item():.0f}'
        )


def test_add_scale_dyn_rejects_invalid_scale_max():
    """Reject a non-numeric, non-positive, non-finite, or unreachable scale_max by exact message."""
    grid = 2.0 ** (-FRACWIDTH)
    not_numeric = 'legal values: a positive finite int or float.'
    cases = [
        (0, f'Invalid scale_max: <0>; {not_numeric}'),
        (-1.5, f'Invalid scale_max: <-1.5>; {not_numeric}'),
        (True, f'Invalid scale_max: <True>; {not_numeric}'),
        (float('nan'), f'Invalid scale_max: <nan>; {not_numeric}'),
        # Exactly half a raw unit ties down to zero raw units.
        (0.03125, f'Invalid scale_max: <0.03125>; legal values: greater than half of <{grid}>.'),
    ]

    for scale_max, message in cases:
        try:
            add_scale_dyn({
                'polarity': 'unipolar',
                'scale_max': scale_max,
                'intwidth': INTWIDTH,
                'fracwidth': FRACWIDTH,
            })
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'add_scale_dyn must reject a scale_max of {scale_max}')

    # 2 ** (4 + 2 - 1) - 1 = 31 raw units is 7.75 in value units.
    try:
        add_scale_dyn({
            'polarity': 'unipolar',
            'scale_max': 8,
            'intwidth': 4,
            'fracwidth': 2,
        })
    except AssertionError as error:
        assert str(error) == (
            'add_scale_dyn scale_max <8.0> exceeds accumulator maximum <7.75> for '
            'intwidth <4> and fracwidth <2>.'
        ), str(error)
        return
    raise AssertionError('add_scale_dyn must reject a scale_max above acc_max')


def test_add_scale_dyn_rejects_invalid_accumulator_format():
    """Reject a non-int or out-of-range intwidth or fracwidth by exact message."""
    cases = [
        ('intwidth', 0, 'Invalid intwidth: <0>; legal values: an integer of at least 1.'),
        ('intwidth', 8.0, 'Invalid intwidth: <8.0>; legal values: an integer of at least 1.'),
        ('fracwidth', -1, 'Invalid fracwidth: <-1>; legal values: a non-negative integer.'),
        ('fracwidth', 4.0, 'Invalid fracwidth: <4.0>; legal values: a non-negative integer.'),
    ]

    for key, value, message in cases:
        config = {
            'polarity': 'unipolar',
            'scale_max': SCALE_MAX,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }
        config[key] = value
        try:
            add_scale_dyn(config)
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'add_scale_dyn must reject {key}={value}')


def test_add_scale_dyn_rejects_out_of_range_call_scale():
    """Reject a per-call scale that is non-numeric, non-positive, or above scale_max."""
    input = torch.ones((2, ENTRY), dtype=global_config.stype)
    grid = 2.0 ** (-FRACWIDTH)
    operation = add_scale_dyn({
        'polarity': 'unipolar',
        'scale_max': SCALE_MAX,
        'intwidth': INTWIDTH,
        'fracwidth': FRACWIDTH,
    })

    # A legal call first, so the rejections below land on non-first calls.
    operation(input, 4, dim=-1)

    cases = [
        (0, 'Invalid scale: <0>; legal values: a positive finite int or float.'),
        (-1, 'Invalid scale: <-1>; legal values: a positive finite int or float.'),
        (float('inf'), 'Invalid scale: <inf>; legal values: a positive finite int or float.'),
        (True, 'add_scale_dyn scale must be an int or float scalar: got <True>.'),
        (False, 'add_scale_dyn scale must be an int or float scalar: got <False>.'),
        ('2', 'add_scale_dyn scale must be an int or float scalar: got <2>.'),
        (0.03125, f'Invalid scale: <0.03125>; legal values: greater than half of <{grid}>.'),
        (SCALE_MAX + 1, f'add_scale_dyn scale <{SCALE_MAX + 1}> exceeds scale_max <{float(SCALE_MAX)}>.'),
        (16.05, f'add_scale_dyn scale <16.05> exceeds scale_max <{float(SCALE_MAX)}>.'),
    ]
    for scale, message in cases:
        try:
            operation(input, scale, dim=-1)
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'add_scale_dyn must reject a per-call scale of {scale}')

    # A request rounding down onto scale_max is accepted.
    operation(input, 16.02, dim=-1)
    assert operation.scale == float(SCALE_MAX)


if __name__ == '__main__':
    test_add_scale_dyn()
    test_add_scale_dyn_matches_add_scale_at_constant_scale()
    test_add_scale_dyn_fires_at_the_fractional_scale_rate()
    test_add_scale_dyn_reports_the_effective_call_scale()
    test_add_scale_dyn_conserves_mass_across_a_scale_change()
    test_add_scale_dyn_conserves_mass_across_a_scale_change_bipolar()
    test_add_scale_dyn_saturates_at_the_accumulator_rail()
    test_add_scale_dyn_rejects_invalid_scale_max()
    test_add_scale_dyn_rejects_invalid_accumulator_format()
    test_add_scale_dyn_rejects_out_of_range_call_scale()

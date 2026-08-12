import torch

from napl.sim.base import global_config
from napl.utils._shared_test import devices, streaming_suite
from napl.sim.operation import add_scale, div_scale, div_scale_dyn


SCALE = 2
SCALE_MAX = 4
INTWIDTH = 12
FRACWIDTH = 4
GRID = 2.0 ** (-FRACWIDTH)
SEED = 1234


def _spike_stream(timesteps, shape, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 2, (timesteps,) + shape, generator=generator, dtype=torch.int64
    ).type(global_config.stype)


def make_operation(polarity, _timestep, _device):
    return div_scale_dyn({
        'polarity': polarity,
        'scale_max': SCALE_MAX,
        'intwidth': INTWIDTH,
        'fracwidth': FRACWIDTH,
    })


def make_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(lo, hi, 512).reshape(64, 8),)


def make_random_perf_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(lo, hi, 16384 * 8).reshape(16384, 8),)


def analytic_reference(values, _polarity):
    return values[0] / SCALE


def known_answer_case(_polarity):
    values = torch.ones((8, 8))
    # The divisor is a power of two and every input is the full-scale stream,
    # so the shift is exact and the answer carries no residue.
    return (values,), values / SCALE


CONFIG = {
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'polarities': ['unipolar', 'bipolar'],
    'timesteps': 256,
    'apply_operation': lambda operation, spikes: operation(spikes[0], SCALE),
}


def _shadow_outputs(stream, schedule, fracwidth, acc_min, acc_max, polarity):
    """Replay a div_scale_dyn stream on a doubled-LSB integer accumulator."""
    # Doubling every raw quantity turns the bipolar half-unit offset into an integer, so the
    # whole model stays in exact integer arithmetic.
    accumulator = torch.zeros(stream.shape[1:], dtype=torch.int64)
    outputs = []
    for timestep in range(stream.shape[0]):
        scale_raw = schedule[timestep]
        offset_doubled = (2**fracwidth - scale_raw) if polarity == 'bipolar' else 0
        accumulator.add_(stream[timestep].type(torch.int64), alpha=2**(fracwidth+1))
        accumulator.sub_(offset_doubled)
        accumulator.clamp_(2 * acc_min, 2 * acc_max)
        output = torch.ge(accumulator, 2 * scale_raw).type(torch.int64)
        accumulator.sub_(output, alpha=2 * scale_raw)
        outputs.append(output)
    return outputs, accumulator


def test_div_scale_dyn():
    """Verify div_scale_dyn at a held scale against analytic and known-answer streams."""
    streaming_suite(CONFIG)
    print('Test passed.')


def test_div_scale_dyn_fires_at_the_fractional_scale_rate():
    """Verify a fractional per-call scale divides an all-ones stream at the exact target rate."""
    timesteps = 60
    scale = 1.5
    input_cpu = torch.ones((4, 3), dtype=global_config.stype)
    # An all-ones stream is p_in = 1 (v_in = 1), so the target rates are 1 / scale
    # unipolar and (1 + 1 / scale) / 2 bipolar.
    expected_rate = {'unipolar': 1.0 / scale, 'bipolar': (1.0 + 1.0 / scale) / 2}

    for device in devices():
        input = input_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            operation = div_scale_dyn({
                'polarity': polarity,
                'scale_max': SCALE_MAX,
                'intwidth': INTWIDTH,
                'fracwidth': FRACWIDTH,
            }).to(device)

            fires = torch.zeros((4, 3), dtype=global_config.ntype, device=device)
            for _ in range(timesteps):
                output = operation(input, scale)
                assert output.shape == (4, 3), (
                    f'[{device}] {polarity} output shape {tuple(output.shape)} != (4, 3)'
                )
                fires.add_(output.type(global_config.ntype))

            rate = fires / timesteps
            expected = torch.full_like(rate, expected_rate[polarity])
            torch.testing.assert_close(rate, expected, atol=1e-6, rtol=0)
            print(
                f'[{device}][{polarity}] scale={scale}, '
                f'rate={rate[0, 0].item():.6f}, '
                f'expected={expected_rate[polarity]:.6f}'
            )


def test_div_scale_dyn_matches_div_scale_at_constant_scale():
    """Verify div_scale_dyn reproduces div_scale bit-exactly when the scale never changes."""
    timesteps = 128
    stream_cpu = _spike_stream(timesteps, (16, 4))

    compared = 0
    for device in devices():
        stream = stream_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            for scale in (0.5, 1, 1.5, SCALE_MAX):
                reference = div_scale({
                    'polarity': polarity,
                    'scale': scale,
                    'intwidth': INTWIDTH,
                    'fracwidth': FRACWIDTH,
                }).to(device)
                candidate = div_scale_dyn({
                    'polarity': polarity,
                    'scale_max': SCALE_MAX,
                    'intwidth': INTWIDTH,
                    'fracwidth': FRACWIDTH,
                }).to(device)
                for timestep in range(timesteps):
                    expected = reference(stream[timestep])
                    observed = candidate(stream[timestep], scale)
                    assert torch.equal(observed, expected), (
                        f'[{device}] {polarity} scale={scale} diverged at '
                        f'timestep {timestep}'
                    )
                assert torch.equal(candidate.accumulator, reference.accumulator)
                compared += 1
        print(f'[{device}] div_scale_dyn matched div_scale on 8 polarity/scale pairs')

    assert compared == 8 * len(devices())


def test_div_scale_dyn_matches_add_scale_at_integer_scale():
    """Verify div_scale_dyn reproduces add_scale with a single entry when the scale is an integer."""
    timesteps = 128
    width = 12
    stream_cpu = _spike_stream(timesteps, (16, 4), seed=SEED + 2)

    for device in devices():
        stream = stream_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            for scale in (1, 2, 4):
                # At fracwidth f the accumulator counts 2 ** f raw units per add_scale unit, so the
                # spikes must still match and the accumulator must carry that exact factor.
                for fracwidth in (0, FRACWIDTH):
                    unit = 2**fracwidth
                    reference = add_scale({
                        'polarity': polarity, 'scale': scale,
                        'intwidth': width, 'fracwidth': 0,
                    }).to(device)
                    candidate = div_scale_dyn({
                        'polarity': polarity,
                        'scale_max': SCALE_MAX,
                        'intwidth': width,
                        'fracwidth': fracwidth,
                    }).to(device)
                    for timestep in range(timesteps):
                        # entry=1 with dim=None is add_scale's single-stream form.
                        expected = reference(stream[timestep], entry=1, dim=None)
                        observed = candidate(stream[timestep], scale)
                        assert torch.equal(observed, expected), (
                            f'[{device}] {polarity} scale={scale} '
                            f'fracwidth={fracwidth} diverged at timestep {timestep}'
                        )
                    assert torch.equal(
                        candidate.accumulator, reference.accumulator * unit
                    ), (
                        f'[{device}] {polarity} scale={scale} fracwidth={fracwidth} '
                        f'accumulator {candidate.accumulator.tolist()} != '
                        f'{unit} * {reference.accumulator.tolist()}'
                    )
        print(f'[{device}] div_scale_dyn matched add_scale on 12 polarity/scale/fracwidth cases')


def test_div_scale_dyn_matches_a_doubled_lsb_shadow_at_odd_scale_raw():
    """Verify bipolar half-unit accumulator states match an exact doubled-LSB integer model."""
    timesteps = 128
    fracwidth = 4
    stream_cpu = _spike_stream(timesteps, (16, 4), seed=SEED + 3)
    # 2 ** fracwidth - scale_raw is odd for each of these, so the bipolar offset is a half raw
    # unit and the accumulator visits half-unit states.
    odd_scale_raw = (17, 3, 31)
    schedule = [odd_scale_raw[step % len(odd_scale_raw)] for step in range(timesteps)]

    for device in devices():
        stream = stream_cpu.to(device)
        operation = div_scale_dyn({
            'polarity': 'bipolar',
            'scale_max': SCALE_MAX,
            'intwidth': INTWIDTH,
            'fracwidth': fracwidth,
        }).to(device)
        expected_outputs, expected_accumulator = _shadow_outputs(
            stream_cpu, schedule, fracwidth,
            operation.acc_min, operation.acc_max, 'bipolar',
        )

        half_unit_steps = 0
        for timestep in range(timesteps):
            observed = operation(stream[timestep], schedule[timestep] / 2**fracwidth)
            assert torch.equal(
                observed.cpu().type(torch.int64), expected_outputs[timestep]
            ), (
                f'[{device}] scale_raw={schedule[timestep]} diverged at timestep '
                f'{timestep}: {observed.tolist()} != '
                f'{expected_outputs[timestep].tolist()}'
            )
            if (operation.accumulator.cpu() % 1 != 0).any().item():
                half_unit_steps += 1
        doubled = (operation.accumulator.cpu() * 2).type(torch.int64)
        assert torch.equal(doubled, expected_accumulator), (
            f'[{device}] doubled accumulator {doubled.tolist()} != '
            f'{expected_accumulator.tolist()}'
        )
        assert half_unit_steps > 0, (
            f'[{device}] never reached a half-unit accumulator state'
        )
        print(
            f'[{device}] matched the doubled-LSB shadow over {timesteps} timesteps '
            f'of alternating odd scale_raw, {half_unit_steps} of them at half-unit '
            f'accumulator states'
        )


def test_div_scale_dyn_saturates_at_the_accumulator_rail():
    """Verify a narrow accumulator clamps at acc_max before the carry is subtracted."""
    timesteps = 100
    intwidth = 6
    fracwidth = 2
    scale = 0.5
    input_cpu = torch.ones((4,), dtype=global_config.stype)

    for device in devices():
        input = input_cpu.to(device)
        operation = div_scale_dyn({
            'polarity': 'unipolar',
            'scale_max': SCALE_MAX,
            'intwidth': intwidth,
            'fracwidth': fracwidth,
        }).to(device)

        for _ in range(timesteps):
            output = operation(input, scale)

        # Inflow of one unit per timestep outruns the drain of scale, so the clamp holds the
        # accumulator at acc_max and the carry leaves acc_max - scale_raw.
        scale_raw = round(scale * 2**fracwidth)
        railed = operation.acc_max - scale_raw
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


def test_div_scale_dyn_conserves_mass_across_a_scale_change():
    """Verify fired mass plus the accumulator equals the input mass across a scale change."""
    timesteps = 64
    # The off-grid late scale quantizes to 1.25, and the mass below is accounted at that
    # effective value rather than at the request.
    early_scale, late_scale = 2.0, 1.23
    unit = 2 ** FRACWIDTH
    stream_cpu = _spike_stream(timesteps, (4, 4), seed=SEED + 1)

    for device in devices():
        stream = stream_cpu.to(device)
        operation = div_scale_dyn({
            'polarity': 'unipolar',
            'scale_max': SCALE_MAX,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }).to(device)

        fired_mass = torch.zeros((4, 4), dtype=global_config.ntype, device=device)
        for timestep in range(timesteps):
            scale = early_scale if timestep < timesteps // 2 else late_scale
            output = operation(stream[timestep], scale)
            fired_mass.add_(
                output.type(global_config.ntype), alpha=round(scale * unit)
            )

        total_input = stream.type(global_config.ntype).sum(dim=0) * unit
        residual = total_input - fired_mass

        assert torch.equal(residual, operation.accumulator), (
            f'[{device}] residual {residual.tolist()} != accumulator '
            f'{operation.accumulator.tolist()}'
        )
        assert residual.min().item() >= 0, f'[{device}] negative residual'
        assert residual.max().item() < operation.acc_max, (
            f'[{device}] accumulator clamped'
        )
        print(
            f'[{device}] input={total_input.sum().item():.0f}, '
            f'fired={fired_mass.sum().item():.0f}, '
            f'residual max={residual.max().item():.0f} raw units'
        )


def test_div_scale_dyn_conserves_mass_across_a_scale_change_bipolar():
    """Verify the bipolar offset tracks the current call scale across a scale change."""
    timesteps = 64
    # The off-grid late scale quantizes to 1.25, and the mass below is accounted at that
    # effective value rather than at the request.
    early_scale, late_scale = 2.0, 1.23
    unit = 2 ** FRACWIDTH
    stream_cpu = _spike_stream(timesteps, (4, 4), seed=SEED + 1)

    for device in devices():
        stream = stream_cpu.to(device)
        operation = div_scale_dyn({
            'polarity': 'bipolar',
            'scale_max': SCALE_MAX,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }).to(device)

        fired_mass = torch.zeros((4, 4), dtype=global_config.ntype, device=device)
        expected_total = torch.zeros((4, 4), dtype=global_config.ntype, device=device)
        for timestep in range(timesteps):
            scale = early_scale if timestep < timesteps // 2 else late_scale
            scale_raw = round(scale * unit)
            expected_total.add_(
                stream[timestep].type(global_config.ntype) * unit
                - (unit - scale_raw) / 2
            )
            output = operation(stream[timestep], scale)
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


def test_div_scale_dyn_reset_and_replay_is_deterministic():
    """Verify reset clears the accumulator and a replayed stream reproduces the outputs."""
    timesteps = 32
    stream_cpu = _spike_stream(timesteps, (8, 4), seed=SEED + 1)
    schedule = [1.5 if step % 2 else 2.0 for step in range(timesteps)]

    for device in devices():
        stream = stream_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            operation = div_scale_dyn({
                'polarity': polarity,
                'scale_max': SCALE_MAX,
                'intwidth': INTWIDTH,
                'fracwidth': FRACWIDTH,
            }).to(device)

            first = [
                operation(stream[step], schedule[step])
                for step in range(timesteps)
            ]
            assert operation.timestep_cur == timesteps
            operation.reset()
            assert operation.timestep_cur == 0
            assert operation.accumulator.abs().sum().item() == 0

            replay = [
                operation(stream[step], schedule[step])
                for step in range(timesteps)
            ]
            assert all(
                torch.equal(one, two) for one, two in zip(first, replay)
            ), f'[{device}] {polarity} replay diverged'


def test_div_scale_dyn_quantizes_an_off_grid_scale():
    """Verify an off-grid per-call scale rounds to the grid and runs as the quantized twin."""
    timesteps = 64
    stream_cpu = _spike_stream(timesteps, (16, 4), seed=SEED + 4)
    # round() is half to even, so 0.09375 and 0.15625 (1.5 and 2.5 raw units) both land on 2.
    # 0.031250001 sits just above the half raw unit that rounds to zero, so it lands on 1.
    quantized = {0.1: 0.125, 0.09375: 0.125, 0.15625: 0.125, 1.03: 1.0, 0.031250001: 0.0625}

    for requested, effective in quantized.items():
        for device in devices():
            stream = stream_cpu.to(device)
            for polarity in ('unipolar', 'bipolar'):
                candidate = div_scale_dyn({
                    'polarity': polarity,
                    'scale_max': SCALE_MAX,
                    'intwidth': INTWIDTH,
                    'fracwidth': FRACWIDTH,
                }).to(device)
                twin = div_scale({
                    'polarity': polarity,
                    'scale': effective,
                    'intwidth': INTWIDTH,
                    'fracwidth': FRACWIDTH,
                }).to(device)
                for timestep in range(timesteps):
                    expected = twin(stream[timestep])
                    observed = candidate(stream[timestep], requested)
                    assert torch.equal(observed, expected), (
                        f'[{device}] {polarity} scale={requested} diverged from '
                        f'the {effective} twin at timestep {timestep}'
                    )
                assert candidate.accumulator.shape == (16, 4)
                assert torch.equal(candidate.accumulator, twin.accumulator)
        print(f'per-call scale={requested} quantized to {effective} and matched its twin')

    # A per-call tie quantizing down onto scale_max is accepted: 4.03125 is 64.5 raw units and
    # half to even lands it on the 64 raw units of scale_max.
    for device in devices():
        stream = stream_cpu.to(device)
        candidate = div_scale_dyn({
            'polarity': 'unipolar',
            'scale_max': SCALE_MAX,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }).to(device)
        twin = div_scale({
            'polarity': 'unipolar',
            'scale': float(SCALE_MAX),
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }).to(device)
        for timestep in range(timesteps):
            expected = twin(stream[timestep])
            observed = candidate(stream[timestep], 4.03125)
            assert torch.equal(observed, expected), (
                f'[{device}] per-call scale=4.03125 diverged from the '
                f'{float(SCALE_MAX)} twin at timestep {timestep}'
            )
        assert torch.equal(candidate.accumulator, twin.accumulator)

    # scale_max takes the same quantization, reported through the effective value.
    operation = div_scale_dyn({
        'polarity': 'unipolar',
        'scale_max': 1.03,
        'intwidth': INTWIDTH,
        'fracwidth': FRACWIDTH,
    })
    assert operation.scale_max == 1.0, operation.scale_max
    assert operation.scale_max_raw == 2**FRACWIDTH, operation.scale_max_raw


def test_div_scale_dyn_rejects_out_of_range_call_scale():
    """Reject a per-call scale that is non-scalar, out of range, or rounds to zero by exact message."""
    input = torch.ones((2, 4), dtype=global_config.stype)
    operation = div_scale_dyn({
        'polarity': 'unipolar',
        'scale_max': SCALE_MAX,
        'intwidth': INTWIDTH,
        'fracwidth': FRACWIDTH,
    })

    # A legal call first, so the rejections below land on non-first calls.
    operation(input, 1.5)

    not_numeric = 'legal values: a positive finite int or float.'
    tensor_scale = torch.tensor(1.5)
    cases = [
        (True, 'div_scale_dyn scale must be an int or float scalar: got <True>.'),
        (False, 'div_scale_dyn scale must be an int or float scalar: got <False>.'),
        (
            tensor_scale,
            f'div_scale_dyn scale must be an int or float scalar: got <{tensor_scale}>.',
        ),
        (0, f'Invalid scale: <0>; {not_numeric}'),
        (-1.5, f'Invalid scale: <-1.5>; {not_numeric}'),
        (float('inf'), f'Invalid scale: <inf>; {not_numeric}'),
        (float('nan'), f'Invalid scale: <nan>; {not_numeric}'),
        (0.01, f'Invalid scale: <0.01>; legal values: greater than half of <{GRID}>.'),
        # Exactly half a raw unit ties down to zero raw units.
        (0.03125, f'Invalid scale: <0.03125>; legal values: greater than half of <{GRID}>.'),
        (
            SCALE_MAX + 1,
            f'div_scale_dyn scale <{SCALE_MAX + 1}> exceeds scale_max <{float(SCALE_MAX)}>.',
        ),
    ]

    for scale, message in cases:
        try:
            operation(input, scale)
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'div_scale_dyn must reject a per-call scale of {scale}')


def test_div_scale_dyn_rejects_invalid_config():
    """Reject a rounds-to-zero, unreachable, or malformed configuration by exact message."""
    not_numeric = 'legal values: a positive finite int or float.'
    cases = [
        (
            {'scale_max': 0.01},
            f'Invalid scale_max: <0.01>; legal values: greater than half of <{GRID}>.',
        ),
        (
            # Exactly half a raw unit ties down to zero raw units.
            {'scale_max': 0.03125},
            f'Invalid scale_max: <0.03125>; legal values: greater than half of <{GRID}>.',
        ),
        (
            {'scale_max': 0},
            f'Invalid scale_max: <0>; {not_numeric}',
        ),
        (
            {'scale_max': True},
            f'Invalid scale_max: <True>; {not_numeric}',
        ),
        (
            {'scale_max': '2'},
            f'Invalid scale_max: <2>; {not_numeric}',
        ),
        (
            {'scale_max': float('inf')},
            f'Invalid scale_max: <inf>; {not_numeric}',
        ),
        (
            {'scale_max': float('nan')},
            f'Invalid scale_max: <nan>; {not_numeric}',
        ),
        (
            {'intwidth': 0},
            'Invalid intwidth: <0>; legal values: an integer of at least 1.',
        ),
        (
            {'fracwidth': -1},
            'Invalid fracwidth: <-1>; legal values: a non-negative integer.',
        ),
        (
            {'scale_max': 8, 'intwidth': 4, 'fracwidth': 2},
            'div_scale_dyn scale_max <8.0> exceeds accumulator maximum <7.75> for '
            'intwidth <4> and fracwidth <2>.',
        ),
    ]

    for override, message in cases:
        config = {
            'polarity': 'unipolar',
            'scale_max': SCALE_MAX,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }
        config.update(override)
        try:
            div_scale_dyn(config)
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'div_scale_dyn must reject {override}')


if __name__ == '__main__':
    test_div_scale_dyn()
    test_div_scale_dyn_fires_at_the_fractional_scale_rate()
    test_div_scale_dyn_matches_div_scale_at_constant_scale()
    test_div_scale_dyn_matches_add_scale_at_integer_scale()
    test_div_scale_dyn_matches_a_doubled_lsb_shadow_at_odd_scale_raw()
    test_div_scale_dyn_saturates_at_the_accumulator_rail()
    test_div_scale_dyn_conserves_mass_across_a_scale_change()
    test_div_scale_dyn_conserves_mass_across_a_scale_change_bipolar()
    test_div_scale_dyn_reset_and_replay_is_deterministic()
    test_div_scale_dyn_quantizes_an_off_grid_scale()
    test_div_scale_dyn_rejects_out_of_range_call_scale()
    test_div_scale_dyn_rejects_invalid_config()

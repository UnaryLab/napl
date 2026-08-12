import torch

from napl.sim.base import global_config
from napl.utils._shared_test import devices, streaming_suite
from napl.sim.operation import add_scale, div_scale


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
    return div_scale({
        'polarity': polarity,
        'scale': SCALE,
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
    'apply_operation': lambda operation, spikes: operation(spikes[0]),
}


def _shadow_outputs(stream, scale_raw, fracwidth, acc_min, acc_max, polarity):
    """Replay a div_scale stream on a doubled-LSB integer accumulator."""
    # Doubling every raw quantity turns the bipolar half-unit offset into an integer, so the
    # whole model stays in exact integer arithmetic.
    offset_doubled = (2**fracwidth - scale_raw) if polarity == 'bipolar' else 0
    accumulator = torch.zeros(stream.shape[1:], dtype=torch.int64)
    outputs = []
    for timestep in range(stream.shape[0]):
        accumulator.add_(stream[timestep].type(torch.int64), alpha=2**(fracwidth+1))
        accumulator.sub_(offset_doubled)
        accumulator.clamp_(2 * acc_min, 2 * acc_max)
        output = torch.ge(accumulator, 2 * scale_raw).type(torch.int64)
        accumulator.sub_(output, alpha=2 * scale_raw)
        outputs.append(output)
    return outputs, accumulator


def test_div_scale():
    """Verify div_scale for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)
    print('Test passed.')


def test_div_scale_fires_at_the_fractional_scale_rate():
    """Verify a fractional scale divides an all-ones stream at the exact target rate."""
    timesteps = 60
    scale = 1.5
    input_cpu = torch.ones((4, 3), dtype=global_config.stype)
    # An all-ones stream is p_in = 1 (v_in = 1), so the target rates are 1 / scale
    # unipolar and (1 + 1 / scale) / 2 bipolar.
    expected_rate = {'unipolar': 1.0 / scale, 'bipolar': (1.0 + 1.0 / scale) / 2}

    for device in devices():
        input = input_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            operation = div_scale({
                'polarity': polarity,
                'scale': scale,
                'intwidth': INTWIDTH,
                'fracwidth': FRACWIDTH,
            }).to(device)

            fires = torch.zeros((4, 3), dtype=global_config.ntype, device=device)
            for _ in range(timesteps):
                output = operation(input)
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


def test_div_scale_matches_add_scale_at_integer_scale():
    """Verify div_scale reproduces add_scale with a single entry when the scale is an integer."""
    timesteps = 128
    width = 12
    stream_cpu = _spike_stream(timesteps, (16, 4))

    compared = 0
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
                    candidate = div_scale({
                        'polarity': polarity,
                        'scale': scale,
                        'intwidth': width,
                        'fracwidth': fracwidth,
                    }).to(device)
                    for timestep in range(timesteps):
                        # entry=1 with dim=None is add_scale's single-stream form.
                        expected = reference(stream[timestep], entry=1, dim=None)
                        observed = candidate(stream[timestep])
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
                    compared += 1
        print(f'[{device}] div_scale matched add_scale on 12 polarity/scale/fracwidth cases')

    assert compared == 12 * len(devices())


def test_div_scale_matches_a_doubled_lsb_shadow_at_odd_scale_raw():
    """Verify bipolar half-unit accumulator states match an exact doubled-LSB integer model."""
    timesteps = 128
    fracwidth = 4
    stream_cpu = _spike_stream(timesteps, (16, 4), seed=SEED + 3)

    for device in devices():
        stream = stream_cpu.to(device)
        # 2 ** fracwidth - scale_raw is odd for each of these, so the bipolar offset is a half
        # raw unit and the accumulator visits half-unit states.
        for scale_raw in (17, 3, 31):
            scale = scale_raw / 2**fracwidth
            operation = div_scale({
                'polarity': 'bipolar',
                'scale': scale,
                'intwidth': INTWIDTH,
                'fracwidth': fracwidth,
            }).to(device)
            expected_outputs, expected_accumulator = _shadow_outputs(
                stream_cpu, scale_raw, fracwidth,
                operation.acc_min, operation.acc_max, 'bipolar',
            )

            half_unit_steps = 0
            for timestep in range(timesteps):
                observed = operation(stream[timestep])
                assert torch.equal(
                    observed.cpu().type(torch.int64), expected_outputs[timestep]
                ), (
                    f'[{device}] scale_raw={scale_raw} diverged at timestep '
                    f'{timestep}: {observed.tolist()} != '
                    f'{expected_outputs[timestep].tolist()}'
                )
                if (operation.accumulator.cpu() % 1 != 0).any().item():
                    half_unit_steps += 1
            doubled = (operation.accumulator.cpu() * 2).type(torch.int64)
            assert torch.equal(doubled, expected_accumulator), (
                f'[{device}] scale_raw={scale_raw} doubled accumulator '
                f'{doubled.tolist()} != {expected_accumulator.tolist()}'
            )
            assert half_unit_steps > 0, (
                f'[{device}] scale_raw={scale_raw} never reached a half-unit '
                f'accumulator state'
            )
            print(
                f'[{device}] scale_raw={scale_raw} matched the doubled-LSB shadow '
                f'over {timesteps} timesteps, {half_unit_steps} of them at '
                f'half-unit accumulator states'
            )


def test_div_scale_saturates_below_unit_scale():
    """Verify a scale below one rails the accumulator and saturates the output at rate 1."""
    timesteps = 64
    scale = 0.5
    input_cpu = torch.ones((4,), dtype=global_config.stype)

    for device in devices():
        input = input_cpu.to(device)
        operation = div_scale({
            'polarity': 'unipolar',
            'scale': scale,
            'intwidth': 6,
            'fracwidth': 2,
        }).to(device)

        for _ in range(timesteps):
            output = operation(input)

        railed = operation.acc_max - operation.scale_raw
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


def test_div_scale_conserves_mass():
    """Verify fired mass plus the accumulator equals the input mass over a stream."""
    timesteps = 64
    scale = 2.0
    unit = 2 ** FRACWIDTH
    scale_raw = round(scale * unit)
    stream_cpu = _spike_stream(timesteps, (4, 4), seed=SEED + 1)

    for device in devices():
        stream = stream_cpu.to(device)
        operation = div_scale({
            'polarity': 'unipolar',
            'scale': scale,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }).to(device)

        fired_mass = torch.zeros((4, 4), dtype=global_config.ntype, device=device)
        for timestep in range(timesteps):
            output = operation(stream[timestep])
            fired_mass.add_(output.type(global_config.ntype), alpha=scale_raw)

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


def test_div_scale_reset_and_replay_is_deterministic():
    """Verify reset clears the accumulator and a replayed stream reproduces the outputs."""
    timesteps = 32
    stream_cpu = _spike_stream(timesteps, (8, 4), seed=SEED + 1)

    for device in devices():
        stream = stream_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            operation = div_scale({
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


def test_div_scale_quantizes_an_off_grid_scale():
    """Verify an off-grid scale rounds to the grid and runs as the quantized twin."""
    timesteps = 64
    stream_cpu = _spike_stream(timesteps, (16, 4), seed=SEED + 4)
    # round() is half to even, so 0.09375 and 0.15625 (1.5 and 2.5 raw units) both land on 2.
    # 0.031250001 sits just above the half raw unit that rounds to zero, so it lands on 1.
    quantized = {0.1: 0.125, 0.09375: 0.125, 0.15625: 0.125, 1.03: 1.0, 0.031250001: 0.0625}

    for requested, effective in quantized.items():
        for device in devices():
            stream = stream_cpu.to(device)
            for polarity in ('unipolar', 'bipolar'):
                candidate = div_scale({
                    'polarity': polarity,
                    'scale': requested,
                    'intwidth': INTWIDTH,
                    'fracwidth': FRACWIDTH,
                }).to(device)
                assert candidate.scale == effective, (
                    f'scale {requested} reported as {candidate.scale}, '
                    f'expected {effective}'
                )
                assert candidate.scale_raw == round(effective * 2**FRACWIDTH)
                twin = div_scale({
                    'polarity': polarity,
                    'scale': effective,
                    'intwidth': INTWIDTH,
                    'fracwidth': FRACWIDTH,
                }).to(device)
                for timestep in range(timesteps):
                    expected = twin(stream[timestep])
                    observed = candidate(stream[timestep])
                    assert torch.equal(observed, expected), (
                        f'[{device}] {polarity} scale={requested} diverged from '
                        f'the {effective} twin at timestep {timestep}'
                    )
                assert candidate.accumulator.shape == (16, 4)
                assert torch.equal(candidate.accumulator, twin.accumulator)
        print(f'scale={requested} quantized to {effective} and matched its twin')

    # 2 ** (4 + 2 - 1) - 1 = 31 raw units is 7.75 in value units, and 7.8 quantizes onto it.
    operation = div_scale({
        'polarity': 'unipolar',
        'scale': 7.8,
        'intwidth': 4,
        'fracwidth': 2,
    })
    assert operation.scale == 7.75, operation.scale
    assert operation.scale_raw == operation.acc_max, operation.scale_raw


def test_div_scale_rejects_invalid_scale():
    """Reject a non-numeric, non-positive, non-finite, or unreachable scale by exact message."""
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
        (0.01, f'Invalid scale: <0.01>; {too_small}'),
        # Exactly half a raw unit ties down to zero raw units.
        (0.03125, f'Invalid scale: <0.03125>; {too_small}'),
    ]

    for scale, message in cases:
        try:
            div_scale({
                'polarity': 'unipolar',
                'scale': scale,
                'intwidth': INTWIDTH,
                'fracwidth': FRACWIDTH,
            })
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'div_scale must reject a scale of {scale}')

    # 2 ** (4 + 2 - 1) - 1 = 31 raw units is 7.75 in value units.
    try:
        div_scale({
            'polarity': 'unipolar',
            'scale': 8,
            'intwidth': 4,
            'fracwidth': 2,
        })
    except AssertionError as error:
        assert str(error) == (
            'div_scale scale <8.0> exceeds accumulator maximum <7.75> for '
            'intwidth <4> and fracwidth <2>.'
        ), str(error)
        return
    raise AssertionError('div_scale must reject a scale above acc_max')


def test_div_scale_rejects_invalid_accumulator_format():
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
            'scale': SCALE,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }
        config[key] = value
        try:
            div_scale(config)
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'div_scale must reject {key}={value}')


if __name__ == '__main__':
    test_div_scale()
    test_div_scale_fires_at_the_fractional_scale_rate()
    test_div_scale_matches_add_scale_at_integer_scale()
    test_div_scale_matches_a_doubled_lsb_shadow_at_odd_scale_raw()
    test_div_scale_saturates_below_unit_scale()
    test_div_scale_conserves_mass()
    test_div_scale_reset_and_replay_is_deterministic()
    test_div_scale_quantizes_an_off_grid_scale()
    test_div_scale_rejects_invalid_scale()
    test_div_scale_rejects_invalid_accumulator_format()

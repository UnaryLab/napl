import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import add_scale, decode, encode
from napl.sim.metric import accuracy


INTWIDTH = 20
FRACWIDTH = 4
SEED = 1234


def _spike_stream(timesteps, shape, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 2, (timesteps,) + shape, generator=generator, dtype=torch.int64
    ).type(global_config.stype)


def _shadow_outputs(stream, entry, scale_raw, fracwidth, acc_min, acc_max, polarity):
    """Replay an add_scale stream on a doubled-LSB integer accumulator."""
    # Doubling every raw quantity turns the bipolar half-unit offset into an integer, so the
    # whole model stays in exact integer arithmetic.
    offset_doubled = (entry * 2**fracwidth - scale_raw) if polarity == 'bipolar' else 0
    accumulator = torch.zeros(stream.shape[1:-1], dtype=torch.int64)
    outputs = []
    for timestep in range(stream.shape[0]):
        accumulator.add_(
            stream[timestep].type(torch.int64).sum(dim=-1), alpha=2**(fracwidth+1)
        )
        accumulator.sub_(offset_doubled)
        accumulator.clamp_(2 * acc_min, 2 * acc_max)
        output = torch.ge(accumulator, 2 * scale_raw).type(torch.int64)
        accumulator.sub_(output, alpha=2 * scale_raw)
        outputs.append(output)
    return outputs, accumulator


class napl_add_scale(napl_base):
    def __init__(self, codec_config, add_scale_config):
        super().__init__()
        self.encoder = encode(codec_config)
        self.decoder = decode(codec_config)
        self.accuracy = accuracy({'polarity': codec_config['polarity']})
        self.add_scale = add_scale(add_scale_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.add_scale(i_spike, dim=-1)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def _kernel_specific_checks():
    """
    Test add_scale with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    add_scale_config={
        'polarity': 'bipolar',
        'scale': 128,
        'intwidth': INTWIDTH,
        'fracwidth': FRACWIDTH,
    }

    input_cpu = gen_rand_tensor(
        codec_config['polarity'],
        shape=(10000, add_scale_config['scale']),
        width=math.log2(codec_config['timestep']),
    ).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        add_scale_inst = napl_add_scale(codec_config, add_scale_config).to(device)

        with timer(device) as elapsed:
            add_scale_inst(input, timesteps=codec_config['timestep'])

        r_value = torch.sum(input, dim=-1) / add_scale_config['scale']
        error, _ = add_scale_inst.accuracy.analyze(r_value, verbose=True)
        max_error = error.abs().max().item()
        assert add_scale_inst.add_scale.timestep_cur == codec_config['timestep']
        add_scale_inst.reset()
        assert add_scale_inst.add_scale.timestep_cur == 0
        print(f'[{device}] max_error={max_error:.4f}, time={elapsed.seconds:.3f}s')

    print('Test passed.')


def make_operation(polarity, _timestep, _device):
    return add_scale({
        'polarity': polarity,
        'scale': 8,
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
    return values[0].mean(dim=-1)


def known_answer_case(_polarity):
    values = torch.ones((8, 8))
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
    'apply_operation': lambda operation, spikes: operation(spikes[0], dim=-1),
    'extra_checks': _kernel_specific_checks,
}


def test_add_scale():
    """Verify add_scale for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


def test_add_scale_emits_carry_on_inclusive_threshold():
    """Verify add_scale emits a carry when the accumulator reaches the scale."""
    input = torch.ones((2, 2), dtype=global_config.stype)

    for polarity in ('unipolar', 'bipolar'):
        for fracwidth in (0, FRACWIDTH):
            operation = add_scale({
                'polarity': polarity,
                'scale': 2,
                'intwidth': 8,
                'fracwidth': fracwidth,
            })
            output = operation(input, dim=-1)
            assert torch.equal(output, torch.ones(2, dtype=global_config.stype)), (
                f'{polarity} fracwidth={fracwidth} did not carry on the threshold'
            )


def test_add_scale_fires_at_the_fractional_scale_rate():
    """Verify a fractional scale reduces an all-ones stream at the exact target rate."""
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
            operation = add_scale({
                'polarity': polarity,
                'scale': scale,
                'intwidth': INTWIDTH,
                'fracwidth': FRACWIDTH,
            }).to(device)

            fires = torch.zeros((4,), dtype=global_config.ntype, device=device)
            for _ in range(timesteps):
                output = operation(input, dim=-1)
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


def test_add_scale_matches_a_doubled_lsb_shadow():
    """Verify both polarities match an exact doubled-LSB integer model at fracwidth 0 and 4."""
    timesteps = 128
    entry = 8
    intwidth = 12
    stream_cpu = _spike_stream(timesteps, (16, entry), seed=SEED + 2)

    for device in devices():
        stream = stream_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            # fracwidth 0 reproduces the integer accumulator, and fracwidth 4 with an odd
            # entry * 2 ** fracwidth - scale_raw drives the accumulator onto half raw units.
            for fracwidth, scale in ((0, 3), (FRACWIDTH, 3), (FRACWIDTH, 2.5), (FRACWIDTH, 1.9375)):
                operation = add_scale({
                    'polarity': polarity,
                    'scale': scale,
                    'intwidth': intwidth,
                    'fracwidth': fracwidth,
                }).to(device)
                expected_outputs, expected_accumulator = _shadow_outputs(
                    stream_cpu, entry, operation.scale_raw, fracwidth,
                    operation.acc_min, operation.acc_max, polarity,
                )

                for timestep in range(timesteps):
                    observed = operation(stream[timestep], dim=-1)
                    assert torch.equal(
                        observed.cpu().type(torch.int64), expected_outputs[timestep]
                    ), (
                        f'[{device}] {polarity} fracwidth={fracwidth} scale={scale} '
                        f'diverged at timestep {timestep}'
                    )
                doubled = (operation.accumulator.cpu() * 2).type(torch.int64)
                assert torch.equal(doubled, expected_accumulator), (
                    f'[{device}] {polarity} fracwidth={fracwidth} scale={scale} '
                    f'doubled accumulator {doubled.tolist()} != '
                    f'{expected_accumulator.tolist()}'
                )
        print(
            f'[{device}] add_scale matched the doubled-LSB shadow on 8 '
            f'polarity/fracwidth/scale cases over {timesteps} timesteps'
        )


def test_add_scale_saturates_below_the_fan_in_scale():
    """Verify a scale below the fan-in rails the accumulator and saturates the output at rate 1."""
    timesteps = 64
    entry = 4
    scale = 0.5
    input_cpu = torch.ones((4, entry), dtype=global_config.stype)

    for device in devices():
        input = input_cpu.to(device)
        operation = add_scale({
            'polarity': 'unipolar',
            'scale': scale,
            'intwidth': 6,
            'fracwidth': 2,
        }).to(device)

        for _ in range(timesteps):
            output = operation(input, dim=-1)

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


def test_add_scale_conserves_mass():
    """Verify fired mass plus the accumulator equals the reduced input mass over a stream."""
    timesteps = 64
    entry = 8
    scale = 2.5
    unit = 2 ** FRACWIDTH
    scale_raw = round(scale * unit)
    stream_cpu = _spike_stream(timesteps, (4, entry), seed=SEED + 1)

    for device in devices():
        stream = stream_cpu.to(device)
        operation = add_scale({
            'polarity': 'unipolar',
            'scale': scale,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }).to(device)

        fired_mass = torch.zeros((4,), dtype=global_config.ntype, device=device)
        for timestep in range(timesteps):
            output = operation(stream[timestep], dim=-1)
            fired_mass.add_(output.type(global_config.ntype), alpha=scale_raw)

        total_input = stream.type(global_config.ntype).sum(dim=(0, -1)) * unit
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


def test_add_scale_reset_and_replay_is_deterministic():
    """Verify reset clears the accumulator and a replayed stream reproduces the outputs."""
    timesteps = 32
    stream_cpu = _spike_stream(timesteps, (8, 4), seed=SEED + 1)

    for device in devices():
        stream = stream_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            operation = add_scale({
                'polarity': polarity,
                'scale': 1.5,
                'intwidth': INTWIDTH,
                'fracwidth': FRACWIDTH,
            }).to(device)

            first = [operation(stream[step], dim=-1) for step in range(timesteps)]
            assert operation.timestep_cur == timesteps
            operation.reset()
            assert operation.timestep_cur == 0
            assert operation.accumulator.abs().sum().item() == 0
            assert operation.is_first_call

            replay = [operation(stream[step], dim=-1) for step in range(timesteps)]
            assert all(
                torch.equal(one, two) for one, two in zip(first, replay)
            ), f'[{device}] {polarity} replay diverged'


def test_add_scale_quantizes_an_off_grid_scale():
    """Verify an off-grid scale rounds to the grid and runs as the quantized twin."""
    timesteps = 64
    stream_cpu = _spike_stream(timesteps, (16, 8), seed=SEED + 4)
    # round() is half to even, so 0.09375 and 0.15625 (1.5 and 2.5 raw units) both land on 2.
    # 0.031250001 sits just above the half raw unit that rounds to zero, so it lands on 1.
    quantized = {0.1: 0.125, 0.09375: 0.125, 0.15625: 0.125, 1.03: 1.0, 0.031250001: 0.0625}

    for requested, effective in quantized.items():
        for device in devices():
            stream = stream_cpu.to(device)
            for polarity in ('unipolar', 'bipolar'):
                candidate = add_scale({
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
                twin = add_scale({
                    'polarity': polarity,
                    'scale': effective,
                    'intwidth': INTWIDTH,
                    'fracwidth': FRACWIDTH,
                }).to(device)
                for timestep in range(timesteps):
                    expected = twin(stream[timestep], dim=-1)
                    observed = candidate(stream[timestep], dim=-1)
                    assert torch.equal(observed, expected), (
                        f'[{device}] {polarity} scale={requested} diverged from '
                        f'the {effective} twin at timestep {timestep}'
                    )
                assert candidate.accumulator.shape == (16,)
                assert torch.equal(candidate.accumulator, twin.accumulator)
        print(f'scale={requested} quantized to {effective} and matched its twin')

    # 2 ** (4 + 2 - 1) - 1 = 31 raw units is 7.75 in value units, and 7.8 quantizes onto it.
    operation = add_scale({
        'polarity': 'unipolar',
        'scale': 7.8,
        'intwidth': 4,
        'fracwidth': 2,
    })
    assert operation.scale == 7.75, operation.scale
    assert operation.scale_raw == operation.acc_max, operation.scale_raw


def test_add_scale_rejects_invalid_scale():
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
            add_scale({
                'polarity': 'unipolar',
                'scale': scale,
                'intwidth': INTWIDTH,
                'fracwidth': FRACWIDTH,
            })
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'add_scale must reject a scale of {scale}')

    # 2 ** (4 + 2 - 1) - 1 = 31 raw units is 7.75 in value units.
    try:
        add_scale({
            'polarity': 'unipolar',
            'scale': 8,
            'intwidth': 4,
            'fracwidth': 2,
        })
    except AssertionError as error:
        assert str(error) == (
            'add_scale scale <8.0> exceeds accumulator maximum <7.75> for '
            'intwidth <4> and fracwidth <2>.'
        ), str(error)
        return
    raise AssertionError('add_scale must reject a scale above acc_max')


def test_add_scale_rejects_invalid_accumulator_format():
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
            'scale': 2,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }
        config[key] = value
        try:
            add_scale(config)
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'add_scale must reject {key}={value}')


if __name__ == '__main__':
    test_add_scale()
    test_add_scale_emits_carry_on_inclusive_threshold()
    test_add_scale_fires_at_the_fractional_scale_rate()
    test_add_scale_matches_a_doubled_lsb_shadow()
    test_add_scale_saturates_below_the_fan_in_scale()
    test_add_scale_conserves_mass()
    test_add_scale_reset_and_replay_is_deterministic()
    test_add_scale_quantizes_an_off_grid_scale()
    test_add_scale_rejects_invalid_scale()
    test_add_scale_rejects_invalid_accumulator_format()

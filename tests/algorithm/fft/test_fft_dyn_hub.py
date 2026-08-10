"""Test the dynamic-scale FFT wrapper; gradients are exempt because it has no trainable parameters.

The wrapper carries boundary codecs only. The spike comparison inside the
wrapped ``fft_dyn`` is not differentiable and carries no straight-through
estimator, so no gradient reaches anything and gate 6 has no defining equation
to check.
"""

import math

import torch

from napl.sim.algorithm.fft import fft_dyn, fft_dyn_hub
from napl.sim.base import global_config
from napl.sim.operation import decode, encode
from napl.utils._shared_test import benchmark, devices


POINT = 8
TIMESTEP = 2048
SCALE_MAX = 6
SCALE = 2
WIDTH = int(math.log2(TIMESTEP)) + 1

# Fidelity bound, derived. The wrapper is exactly encode -> bare fft_dyn ->
# decode, so its only error source is the rate-coded stream itself and gate 5's
# stochastic-computing bound applies unchanged: relative error 1 / sqrt(N) at
# N = 2048 timesteps, which is 0.022097, scaled by the reference RMS. This is
# the same bound the bare-core test test_fft_dyn.py carries. The chosen coverage
# multiplier on it is 1.0, that is, none: the measured relative errors span
# 0.0036 to 0.0127 across the held-scale cases below, and nothing is added.
# Per case: known-answer impulse 0.0096 and bin-1 tone 0.0065, mixed scales
# 0.0068, streaming 0.0107 at scales=2, 0.0127 at scales=[2, 2, 3], and 0.0036
# at scales=[1, 2, 2].
# The bound holds for the full-range fidelity inputs gate 5 prescribes; below
# roughly half full scale the error becomes a fixed absolute floor of about
# 0.017 to 0.029 that outgrows the reference-RMS-scaled bound, so the streaming
# input at amplitude 0.25 measures relative error 0.0282 and at amplitude 0.05
# measures 0.1272. Callers running small-amplitude inputs must scale the bound
# accordingly.
RELATIVE_ERROR_BOUND = 1 / math.sqrt(TIMESTEP)

# Gain bounds. The derived expectation is exactly 1.0, because the core
# compensation is the integer product this call's stage adders divide by and the
# codec is unbiased. The +/- 0.01 band is a chosen coverage multiplier of about
# 1.7x on the largest deviation observed here (0.0058, on the impulse known
# answer).
GAIN_MIN = 0.99
GAIN_MAX = 1.01


def _configs(dim=1, scale_max=SCALE_MAX, width=WIDTH):
    codec_config = {
        'polarity': 'bipolar',
        'timestep': TIMESTEP,
        'generator': 'sobol',
        'dim': dim,
    }
    mul_config = {
        'polarity': 'bipolar',
        'timestep': TIMESTEP,
        'generator': 'sobol',
    }
    add_config = {
        'polarity': 'bipolar',
        'scale_max': scale_max,
        'width': width,
    }
    return codec_config, mul_config, add_config


def _make_hub(device, dim=1):
    return fft_dyn_hub(POINT, *_configs(dim)).to(device)


def _make_bare(device, dim=1):
    """Return the manual encode, bare core, and decode composition the wrapper holds."""
    codec_config, mul_config, add_config = _configs(dim)
    encoders = [
        encode(dict(codec_config, dim=dim)).to(device),
        encode(dict(codec_config, dim=dim + 1)).to(device),
    ]
    core = fft_dyn(POINT, mul_config, add_config).to(device)
    decoders = [decode(dict(codec_config)).to(device) for _ in range(2)]
    return encoders, core, decoders


def _bare_step(encoders, core, decoders, values, scales):
    """Advance the manual chain one timestep and return its compensated output."""
    spikes = core(*(encoder(value) for encoder, value in zip(encoders, values)), scales)
    for decoder, spike in zip(decoders, spikes):
        decoder(spike)
    return tuple(decoder.spike_value * core.compensation for decoder in decoders)


def _run(operation, values, scales, timesteps=TIMESTEP):
    output = None
    for _ in range(timesteps):
        output = operation(*values, scales)
    return output


def _reference(values):
    spectrum = torch.fft.fft(torch.complex(*values), dim=0)
    return spectrum.real, spectrum.imag


def _metrics(candidate, reference):
    candidate_complex = torch.complex(*candidate)
    reference_complex = torch.complex(*reference)
    reference_rms = reference_complex.abs().pow(2).mean().sqrt()
    rmse = (candidate_complex - reference_complex).abs().pow(2).mean().sqrt()
    gain = candidate_complex.abs().pow(2).mean().sqrt() / reference_rms
    return rmse.item(), (RELATIVE_ERROR_BOUND * reference_rms).item(), gain.item()


def _sample_values(device):
    full_range = torch.linspace(-1, 1, POINT, dtype=global_config.ntype)
    return (
        torch.stack([full_range, full_range.roll(1), full_range.roll(2)], dim=1).to(device),
        torch.stack([full_range.flip(0), full_range.roll(3), full_range], dim=1).to(device),
    )


def test_fft_dyn_hub_streaming():
    """Verify the numeric-port wrapper reaches the analytic spectrum at held runtime scales."""
    for device in devices():
        values = _sample_values(device)
        reference = _reference(values)
        for scales in (SCALE, [2, 2, 3], [1, 2, 2]):
            operation = _make_hub(device)
            assert operation.streaming is True
            assert operation.point == POINT and operation.stages == 3
            assert operation.scale_max == SCALE_MAX
            # The scales arrive per call, so no static scale list is exposed.
            assert not hasattr(operation, 'scales')
            assert isinstance(operation.core, fft_dyn)
            assert [name for name, _ in operation.named_children()] == [
                'encode_real', 'encode_imag', 'core', 'decode_real', 'decode_imag'
            ]
            # Codecs live only at the boundary, so no stage owns an encoder or decoder.
            assert all(
                not hasattr(getattr(operation.core, f'butterfly_stage_{stage}'), 'encoder_x')
                and not hasattr(getattr(operation.core, f'butterfly_stage_{stage}'), 'decoder_y')
                for stage in range(operation.stages)
            )
            assert not torch.equal(operation.encode_real.num_seq, operation.encode_imag.num_seq)
            assert operation.encoding_io == {}

            output = _run(operation, values, scales)
            requested = [scales] * 3 if isinstance(scales, int) else scales
            assert operation.core.compensation == 2 * math.prod(requested)
            assert all(value.shape == values[0].shape for value in output)
            rmse, relative_bound, gain = _metrics(output, reference)
            assert rmse < relative_bound, (
                f'FFT RMSE {rmse:.4f} exceeds relative bound {relative_bound:.4f}'
            )
            assert GAIN_MIN <= gain <= GAIN_MAX, (
                f'FFT gain {gain:.4f} is outside [{GAIN_MIN}, {GAIN_MAX}]'
            )
            assert operation.timestep_cur == TIMESTEP
            assert operation.core.timestep_cur == TIMESTEP
            assert operation.encode_real.timestep_cur == TIMESTEP
            assert operation.decode_real.timestep_cur == TIMESTEP
            print(
                f'[{device}][scales={scales}] rmse={rmse:.4f}, '
                f'relative_bound={relative_bound:.4f}, gain={gain:.4f}'
            )


def test_fft_dyn_hub_matches_bare_composition():
    """Verify the wrapper is bit-exact against a manual encode, bare core, and decode chain."""
    full_range = torch.linspace(-1, 1, POINT, dtype=global_config.ntype)
    for device in devices():
        values = (
            torch.stack([full_range, full_range.roll(1)], dim=1).to(device),
            torch.stack([full_range.flip(0), full_range], dim=1).to(device),
        )
        for scales in (SCALE, [2, 2, 3]):
            operation = _make_hub(device)
            encoders, core, decoders = _make_bare(device)
            for _ in range(TIMESTEP):
                hub_output = operation(*values, scales)
                bare_output = _bare_step(encoders, core, decoders, values, scales)
                assert all(
                    torch.equal(hub_value, bare_value)
                    for hub_value, bare_value in zip(hub_output, bare_output)
                ), f'[{device}][scales={scales}] diverged at timestep {operation.timestep_cur}'
            print(f'[{device}][scales={scales}] bit-exact against the bare composition '
                  f'for {TIMESTEP} timesteps.')


def test_fft_dyn_hub_known_answer():
    """Verify bipolar impulse and complex-tone spectra detect natural-order twiddles."""
    for device in devices():
        impulse_real = torch.zeros(POINT, 2, dtype=global_config.ntype, device=device)
        impulse_real[0].fill_(1)
        impulse_imag = torch.zeros_like(impulse_real)
        impulse_expected = (torch.ones_like(impulse_real), torch.zeros_like(impulse_imag))

        sample = torch.arange(POINT, dtype=global_config.ntype, device=device)
        angle = 2 * math.pi * sample / POINT
        tone_real = angle.cos().view(POINT, 1).repeat(1, 2)
        tone_imag = angle.sin().view(POINT, 1).repeat(1, 2)
        tone_expected_real = torch.zeros_like(tone_real)
        tone_expected_real[1].fill_(POINT)
        tone_expected = (tone_expected_real, torch.zeros_like(tone_imag))

        for name, values, expected in (
            ('impulse', (impulse_real, impulse_imag), impulse_expected),
            ('bin1_tone', (tone_real, tone_imag), tone_expected),
        ):
            output = _run(_make_hub(device), values, SCALE)
            rmse, relative_bound, gain = _metrics(output, expected)
            assert rmse < relative_bound, (
                f'{name} RMSE {rmse:.4f} exceeds relative bound {relative_bound:.4f}'
            )
            assert GAIN_MIN <= gain <= GAIN_MAX, (
                f'{name} gain {gain:.4f} is outside [{GAIN_MIN}, {GAIN_MAX}]'
            )
            print(
                f'[{device}][{name}] rmse={rmse:.4f}, '
                f'relative_bound={relative_bound:.4f}, gain={gain:.4f}'
            )


def test_fft_dyn_hub_mixed_scales():
    """Verify a mixed runtime scale list reaches each stage adder rather than a uniform one."""
    mixed_scales = [2, 2, 3]
    for device in devices():
        operation = _make_hub(device)
        sample = torch.arange(POINT, dtype=global_config.ntype, device=device)
        angle = 2 * math.pi * sample / POINT
        values = (angle.cos().view(POINT, 1), angle.sin().view(POINT, 1))
        expected_real = torch.zeros_like(values[0])
        expected_real[1].fill_(POINT)
        expected = (expected_real, torch.zeros_like(values[1]))

        output = _run(operation, values, mixed_scales)
        child_scales = [
            getattr(operation.core, f'butterfly_stage_{stage}').compensation
            for stage in range(operation.stages)
        ]
        # These wiring checks are load-bearing: the numerical gates also pass
        # if all three stages incorrectly use the uniform scale list [2, 2, 2].
        assert child_scales == [4, 2, 3]
        assert operation.core.compensation == 2 * 2 * 2 * 3
        rmse, relative_bound, gain = _metrics(output, expected)
        assert rmse < relative_bound, (
            f'mixed-scale RMSE {rmse:.4f} exceeds relative bound {relative_bound:.4f}'
        )
        assert GAIN_MIN <= gain <= GAIN_MAX, (
            f'mixed-scale gain {gain:.4f} is outside [{GAIN_MIN}, {GAIN_MAX}]'
        )
        print(
            f'[{device}][mixed_scales={mixed_scales}] child_scales={child_scales}, '
            f'rmse={rmse:.4f}, relative_bound={relative_bound:.4f}, gain={gain:.4f}'
        )


def test_fft_dyn_hub_reset_replay():
    """Verify reset clears every child and an identical replay reproduces each output."""
    timesteps = 64
    for device in devices():
        values = _sample_values(device)
        operation = _make_hub(device)
        first = [tuple(value.clone() for value in operation(*values, SCALE))
                 for _ in range(timesteps)]
        assert operation.timestep_cur == timesteps

        operation.reset()
        assert operation.timestep_cur == 0
        assert operation.core.timestep_cur == 0
        assert operation.core.compensation is None
        assert operation.encode_real.timestep_cur == 0
        assert operation.encode_imag.timestep_cur == 0
        assert operation.decode_real.timestep_cur == 0
        assert operation.decode_imag.timestep_cur == 0
        assert operation.decode_real.spike_count.abs().sum().item() == 0
        assert operation.decode_imag.spike_count.abs().sum().item() == 0
        assert all(
            torch.count_nonzero(
                getattr(operation.core, f'butterfly_stage_{stage}').add_y.accumulator
            ) == 0
            for stage in range(operation.stages)
        )

        replay = [tuple(value.clone() for value in operation(*values, SCALE))
                  for _ in range(timesteps)]
        assert all(
            torch.equal(before, after)
            for before_step, after_step in zip(first, replay)
            for before, after in zip(before_step, after_step)
        )
        print(f'[{device}] reset and replay reproduced {timesteps} outputs.')


def test_fft_dyn_hub_rejects_invalid_config():
    """Verify construction rejects invalid sizes and dynamic adder configuration by message."""
    codec_config, mul_config, add_config = _configs()
    point_message = (
        'Invalid point: <{0!r}>; legal values: a power-of-two integer '
        'greater than or equal to 2.'
    )
    for point in (0, 3, True, '8'):
        try:
            fft_dyn_hub(point, codec_config, mul_config, add_config)
        except AssertionError as error:
            assert str(error) == point_message.format(point), error
        else:
            raise AssertionError(f'fft_dyn_hub accepted invalid point {point!r}')

    missing = dict(add_config)
    missing.pop('scale_max')
    try:
        fft_dyn_hub(POINT, codec_config, mul_config, missing)
    except AssertionError as error:
        assert str(error) == 'Missing key <scale_max> in the dynamic adder configuration.', error
    else:
        raise AssertionError('fft_dyn_hub accepted an adder configuration missing scale_max')

    for scale_max in (0, -1, True, 2.0, '2'):
        try:
            fft_dyn_hub(POINT, codec_config, mul_config, dict(add_config, scale_max=scale_max))
        except AssertionError as error:
            assert str(error) == (
                f'fft_dyn scale_max must be a positive int: got <{scale_max}>.'
            ), error
        else:
            raise AssertionError(f'fft_dyn_hub accepted scale_max {scale_max!r}')

    accepted = "<['dim', 'generator', 'name', 'polarity', 'seed', 'taps', 'timestep']>"
    try:
        fft_dyn_hub(POINT, dict(codec_config, unsupported=True), mul_config, add_config)
    except AssertionError as error:
        assert str(error) == (
            f'Unknown key <unsupported> in the input configuration; '
            f'accepted keys: {accepted}.'
        ), error
    else:
        raise AssertionError('fft_dyn_hub accepted an unknown codec configuration key')

    try:
        fft_dyn_hub(POINT, dict(codec_config, polarity='unipolar'),
                    dict(mul_config, polarity='unipolar'),
                    dict(add_config, polarity='unipolar'))
    except AssertionError as error:
        assert str(error) == "Invalid polarity: <unipolar>; legal values: <['bipolar']>.", error
    else:
        raise AssertionError('fft_dyn_hub accepted a unipolar configuration')

    # Stage 0 is built for twice scale_max, so the width bound trips there first.
    try:
        fft_dyn_hub(POINT, codec_config, mul_config, dict(add_config, scale_max=10, width=4))
    except AssertionError as error:
        assert str(error) == (
            'FFT stage <0> construction failed: add_any scale <20> exceeds '
            'accumulator maximum <7> for width <4>.'
        ), error
    else:
        raise AssertionError('fft_dyn_hub accepted scale_max 10 above the width-4 maximum')
    print('Test passed.')


def test_fft_dyn_hub_rejects_invalid_scales():
    """Verify invalid runtime scale controls raise before any child advances."""
    operation = _make_hub('cpu')
    values = torch.zeros(POINT, 1, dtype=global_config.ntype)
    operation(values, values, SCALE)
    timesteps = operation.encode_real.timestep_cur
    accumulators = [
        getattr(operation.core, f'butterfly_stage_{stage}').add_y.accumulator.clone()
        for stage in range(operation.stages)
    ]
    cases = [
        (2.0, 'FFT runtime scales must be a Python int or list: got <2.0>.'),
        ('2', "FFT runtime scales must be a Python int or list: got <2>."),
        (True, 'FFT runtime scales must be a Python int or list: got <True>.'),
        ([2, 2], 'FFT runtime scale list length <2> must equal the stage count <3>.'),
        ([2, 2, 2, 2], 'FFT runtime scale list length <4> must equal the stage count <3>.'),
        ([2, 2, 2.0], 'FFT runtime scale at stage <2> must be an int: got <2.0>.'),
        ([2, 2, True], 'FFT runtime scale at stage <2> must be an int: got <True>.'),
        ([2, 2, 0],
         'FFT runtime scale <0> at stage <2> outside the supported range <1> '
         f'to scale_max <{SCALE_MAX}>.'),
        ([2, 2, SCALE_MAX + 1],
         f'FFT runtime scale <{SCALE_MAX + 1}> at stage <2> outside the supported '
         f'range <1> to scale_max <{SCALE_MAX}>.'),
        ([4, 2, 2],
         f'FFT runtime scale <4> at stage <0> must not exceed half of scale_max '
         f'<{SCALE_MAX}>, since stage 0 runs at twice its requested scale.'),
    ]
    for scales, message in cases:
        try:
            operation(values, values, scales)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'fft_dyn_hub accepted runtime scales {scales!r}')
    assert operation.encode_real.timestep_cur == timesteps
    assert operation.encode_imag.timestep_cur == timesteps
    assert operation.core.timestep_cur == timesteps
    assert operation.decode_real.timestep_cur == timesteps
    assert all(
        torch.equal(
            getattr(operation.core, f'butterfly_stage_{stage}').add_y.accumulator,
            accumulators[stage],
        )
        for stage in range(operation.stages)
    )
    print('Test passed.')


def test_fft_dyn_hub_rejects_invalid_shapes():
    """Verify invalid numeric input shapes raise before any child advances."""
    operation = _make_hub('cpu')
    values = torch.zeros(POINT, 1, dtype=global_config.ntype)
    operation(values, values, SCALE)
    timesteps = operation.encode_real.timestep_cur
    for bad, message in (
        ((values, torch.zeros(POINT, 2, dtype=global_config.ntype)),
         'FFT input shapes must match: got <torch.Size([8, 1])> and <torch.Size([8, 2])>.'),
        ((torch.zeros(POINT - 1, 1, dtype=global_config.ntype),) * 2,
         'FFT first input dimension must equal point <8>: got shape <torch.Size([7, 1])>.'),
    ):
        try:
            operation(*bad, SCALE)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(
                f'fft_dyn_hub accepted input shapes {[value.shape for value in bad]}'
            )
    assert operation.encode_real.timestep_cur == timesteps
    assert operation.core.timestep_cur == timesteps
    assert operation.decode_real.timestep_cur == timesteps
    print('Test passed.')


def test_fft_dyn_hub_performance():
    """Measure one timestep per device on a shared input of about 1e5 elements."""
    columns = 12800
    performance_values = (
        torch.linspace(-1, 1, POINT * columns,
                       dtype=global_config.ntype).view(POINT, columns),
        torch.linspace(1, -1, POINT * columns,
                       dtype=global_config.ntype).view(POINT, columns),
    )
    assert performance_values[0].numel() >= 1e5
    cpu_runtime = None
    for device in devices():
        operation = _make_hub(device)
        device_runtime = benchmark(
            lambda values: operation(*values, SCALE),
            performance_values,
            device,
            warmup_runs=1,
            trials=3,
            prepare=operation.reset,
        )
        if device == 'cpu':
            cpu_runtime = device_runtime
        assert cpu_runtime is not None
        print(
            f'[{device}] device_runtime={device_runtime * 1e3:.1f}ms, '
            f'cpu_runtime={cpu_runtime * 1e3:.1f}ms, warmup=1, trials=3, median, '
            f'speedup={cpu_runtime / device_runtime:.2f}x'
        )


if __name__ == '__main__':
    test_fft_dyn_hub_rejects_invalid_config()
    test_fft_dyn_hub_rejects_invalid_scales()
    test_fft_dyn_hub_rejects_invalid_shapes()
    test_fft_dyn_hub_known_answer()
    test_fft_dyn_hub_mixed_scales()
    test_fft_dyn_hub_reset_replay()
    test_fft_dyn_hub_matches_bare_composition()
    test_fft_dyn_hub_streaming()
    test_fft_dyn_hub_performance()
    print('Test passed.')

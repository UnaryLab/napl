"""Test the boundary-codec FFT wrapper; gradients are exempt because it has no trainable parameters.

The spike comparison inside the wrapped ``fft`` is not differentiable and
carries no straight-through estimator, so no gradient reaches anything and
gate 6 has no defining equation to check.
"""

import math

import torch

from napl.sim.algorithm.fft import fft, fft_hub
from napl.sim.base import global_config
from napl.sim.operation import decode, encode
from napl.utils._shared_test import benchmark, devices


POINT = 8
TIMESTEP = 2048
SCALE = 2
WIDTH = int(math.log2(TIMESTEP)) + 1


def _configs(dim=1, scale=SCALE, width=WIDTH):
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
        'scale': scale,
        'intwidth': width,
        'fracwidth': 0,
    }
    return codec_config, mul_config, add_config


def _make_hub(device, dim=1):
    return fft_hub(POINT, *_configs(dim)).to(device)


def _make_bare(device, dim=1):
    """Return the manual encode, bare core, and decode composition the wrapper holds."""
    codec_config, mul_config, add_config = _configs(dim)
    encoders = [
        encode(dict(codec_config, dim=dim)).to(device),
        encode(dict(codec_config, dim=dim + 1)).to(device),
    ]
    core = fft(POINT, mul_config, add_config).to(device)
    decoders = [decode(dict(codec_config)).to(device) for _ in range(2)]
    return encoders, core, decoders


def _bare_step(encoders, core, decoders, values):
    """Advance the manual chain one timestep and return its compensated output."""
    spikes = core(*(encoder(value) for encoder, value in zip(encoders, values)))
    for decoder, spike in zip(decoders, spikes):
        decoder(spike)
    return tuple(decoder.spike_value * core.compensation for decoder in decoders)


def _step_run(operation, values, timesteps=TIMESTEP):
    """Return the final output of an explicit per-timestep run."""
    output = None
    for _ in range(timesteps):
        output = operation.forward_timestep(*values)
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
    return rmse.item(), gain.item()


def _sample_values(device):
    full_range = torch.linspace(-1, 1, POINT, dtype=global_config.ntype)
    return (
        torch.stack([full_range, full_range.roll(1), full_range.roll(2)], dim=1).to(device),
        torch.stack([full_range.flip(0), full_range.roll(3), full_range], dim=1).to(device),
    )


def test_fft_hub_streaming():
    """Verify the numeric-port wrapper reaches the analytic spectrum on every device."""
    for device in devices():
        values = _sample_values(device)
        reference = _reference(values)
        for dim in (1, 2, 3):
            operation = _make_hub(device, dim)
            assert operation.streaming is False
            assert operation.point == POINT and operation.stages == 3
            assert operation.scales == [SCALE] * operation.stages
            assert operation.core.compensation == 2 * SCALE ** operation.stages
            # Every stage multiplier turns its constant twiddle into a stream itself.
            assert operation.internal_encode is True
            # Codecs live only at the boundary, so the core and every stage in it
            # stay in the spike domain and own no encoder or decoder.
            assert isinstance(operation.core, fft)
            assert [name for name, _ in operation.named_children()] == [
                'encode_real', 'encode_imag', 'core', 'decode_real', 'decode_imag'
            ]
            assert all(
                not hasattr(getattr(operation.core, f'butterfly_stage_{stage}'), 'encoder_x')
                and not hasattr(getattr(operation.core, f'butterfly_stage_{stage}'), 'decoder_y')
                for stage in range(operation.stages)
            )
            # The two input streams must decorrelate, so their number sequences differ.
            assert not torch.equal(operation.encode_real.num_seq, operation.encode_imag.num_seq)
            assert operation.encoding_io == {}

            output = operation(*values)
            assert all(value.shape == values[0].shape for value in output)
            rmse, gain = _metrics(output, reference)
            # The wrapper is non-streaming at its numeric interface, so its own
            # counter stays at 0 while the streaming parts it drives count the
            # whole run.
            assert operation.timestep_cur == 0
            assert operation.core.timestep_cur == TIMESTEP
            assert operation.encode_real.timestep_cur == TIMESTEP
            assert operation.decode_real.timestep_cur == TIMESTEP
            print(
                f'[{device}][dim={dim}] rmse={rmse:.4f}, gain={gain:.4f}'
            )


def test_fft_hub_matches_bare_composition():
    """Verify the wrapper is bit-exact against a manual encode, bare core, and decode chain."""
    full_range = torch.linspace(-1, 1, POINT, dtype=global_config.ntype)
    for device in devices():
        values = (
            torch.stack([full_range, full_range.roll(1)], dim=1).to(device),
            torch.stack([full_range.flip(0), full_range], dim=1).to(device),
        )
        operation = _make_hub(device)
        encoders, core, decoders = _make_bare(device)
        for _ in range(TIMESTEP):
            hub_output = operation.forward_timestep(*values)
            bare_output = _bare_step(encoders, core, decoders, values)
            assert all(
                torch.equal(hub_value, bare_value)
                for hub_value, bare_value in zip(hub_output, bare_output)
            ), f'[{device}] diverged at timestep {operation.core.timestep_cur}'
        print(f'[{device}] bit-exact against the bare composition for {TIMESTEP} timesteps.')


def test_fft_hub_matches_timestep_loop():
    """Verify one decorated call equals an explicit timestep loop and repeats bit-exactly."""
    for device in devices():
        values = _sample_values(device)

        looped = _step_run(_make_hub(device), values)

        operation = _make_hub(device)
        single = operation(*values)
        assert all(
            torch.equal(single_value, looped_value)
            for single_value, looped_value in zip(single, looped)
        ), f'[{device}] the decorated call and the {TIMESTEP}-timestep loop disagree'
        assert operation.timestep_cur == 0
        assert operation.core.timestep_cur == TIMESTEP
        assert operation.encode_real.timestep_cur == TIMESTEP
        assert operation.decode_real.timestep_cur == TIMESTEP

        # A decorated call is a fresh run, so a partly advanced wrapper decodes
        # the same spectrum as an untouched one.
        advanced = _make_hub(device)
        advanced.forward_timestep(*values)
        assert all(
            torch.equal(repeated_value, single_value)
            for repeated_value, single_value in zip(advanced(*values), single)
        ), f'[{device}] a repeated decorated call is not a fresh run'
        print(f'[{device}] one call matches the {TIMESTEP}-timestep loop bit-exactly.')


def test_fft_hub_progressive_precision():
    """Verify per-timestep stepping refines the spectrum toward the analytic transform."""
    checkpoints = (TIMESTEP // 8, TIMESTEP)
    for device in devices():
        values = _sample_values(device)
        reference = _reference(values)
        operation = _make_hub(device)
        errors = []
        for timestep in range(1, TIMESTEP + 1):
            output = operation.forward_timestep(*values)
            assert operation.core.timestep_cur == timestep
            if timestep in checkpoints:
                errors.append(_metrics(output, reference)[0])
        assert errors[-1] < errors[0], errors
        print(f'[{device}] rmse refines from {errors[0]:.4f} at N={checkpoints[0]} to '
              f'{errors[-1]:.4f} at N={checkpoints[1]}.')


def test_fft_hub_known_answer():
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
            output = _make_hub(device)(*values)
            rmse, gain = _metrics(output, expected)
            print(
                f'[{device}][{name}] rmse={rmse:.4f}, gain={gain:.4f}'
            )


def test_fft_hub_mixed_scales():
    """Verify mixed stage scales preserve the analytic bipolar complex-tone spectrum."""
    mixed_scales = [2, 2, 3]
    for device in devices():
        codec_config, mul_config, add_config = _configs()
        caller_add_config = dict(add_config, scale=list(mixed_scales))
        original_add_config = dict(caller_add_config)
        operation = fft_hub(POINT, codec_config, mul_config, caller_add_config).to(device)
        assert caller_add_config == original_add_config
        child_scales = [
            getattr(operation.core, f'butterfly_stage_{stage}').add_y.scale
            for stage in range(operation.stages)
        ]
        # These wiring checks are load-bearing: the numerical gates also pass
        # if all three stages incorrectly use the uniform scale list [2, 2, 2].
        assert operation.scales == mixed_scales
        assert child_scales == [4, 2, 3]
        assert operation.core.compensation == 2 * 2 * 2 * 3

        sample = torch.arange(POINT, dtype=global_config.ntype, device=device)
        angle = 2 * math.pi * sample / POINT
        values = (angle.cos().view(POINT, 1), angle.sin().view(POINT, 1))
        expected_real = torch.zeros_like(values[0])
        expected_real[1].fill_(POINT)
        expected = (expected_real, torch.zeros_like(values[1]))
        output = operation(*values)
        rmse, gain = _metrics(output, expected)
        print(
            f'[{device}][mixed_scales={mixed_scales}] child_scales={child_scales}, '
            f'rmse={rmse:.4f}, gain={gain:.4f}'
        )


def test_fft_hub_reset_replay():
    """Verify reset clears every child and an identical replay reproduces each output."""
    timesteps = 64
    for device in devices():
        values = _sample_values(device)
        operation = _make_hub(device)
        first = [tuple(value.clone() for value in operation.forward_timestep(*values))
                 for _ in range(timesteps)]
        assert operation.core.timestep_cur == timesteps

        operation.reset()
        assert operation.timestep_cur == 0
        assert operation.core.timestep_cur == 0
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

        replay = [tuple(value.clone() for value in operation.forward_timestep(*values))
                  for _ in range(timesteps)]
        assert all(
            torch.equal(before, after)
            for before_step, after_step in zip(first, replay)
            for before, after in zip(before_step, after_step)
        )
        print(f'[{device}] reset and replay reproduced {timesteps} outputs.')


def test_fft_hub_rejects_invalid_config():
    """Verify construction rejects invalid sizes, codecs, and scales with exact messages."""
    codec_config, mul_config, add_config = _configs()
    point_message = (
        'Invalid point: <{0!r}>; legal values: a power-of-two integer '
        'greater than or equal to 2.'
    )
    for point in (0, 3, True, '8'):
        try:
            fft_hub(point, codec_config, mul_config, add_config)
        except AssertionError as error:
            assert str(error) == point_message.format(point), error
        else:
            raise AssertionError(f'fft_hub accepted invalid point {point!r}')

    missing = dict(codec_config)
    missing.pop('timestep')
    try:
        fft_hub(POINT, missing, mul_config, add_config)
    except AssertionError as error:
        assert str(error) == 'Missing key <timestep> in the input configuration.', error
    else:
        raise AssertionError('fft_hub accepted a codec configuration missing timestep')

    accepted = "<['dim', 'generator', 'name', 'polarity', 'seed', 'taps', 'timestep']>"
    try:
        fft_hub(POINT, dict(codec_config, unsupported=True), mul_config, add_config)
    except AssertionError as error:
        assert str(error) == (
            f'Unknown key <unsupported> in the input configuration; '
            f'accepted keys: {accepted}.'
        ), error
    else:
        raise AssertionError('fft_hub accepted an unknown codec configuration key')

    missing_add_scale = dict(add_config)
    missing_add_scale.pop('scale')
    try:
        fft_hub(POINT, codec_config, mul_config, missing_add_scale)
    except AssertionError as error:
        assert str(error) == 'Missing key <scale> in the input adder configuration.', error
    else:
        raise AssertionError('fft_hub accepted an adder configuration missing scale')

    try:
        fft_hub(POINT, dict(codec_config, polarity='unipolar'),
                dict(mul_config, polarity='unipolar'),
                dict(add_config, polarity='unipolar'))
    except AssertionError as error:
        assert str(error) == "Invalid polarity: <unipolar>; legal values: <['bipolar']>.", error
    else:
        raise AssertionError('fft_hub accepted a unipolar configuration')

    for scales in ([2, 2], [2, 2, 2, 2]):
        try:
            fft_hub(POINT, codec_config, mul_config, dict(add_config, scale=scales))
        except AssertionError as error:
            assert str(error) == (
                f'FFT scale list length <{len(scales)}> must equal the stage count <3>.'
            ), error
        else:
            raise AssertionError(f'fft_hub accepted scale list with length {len(scales)}')

    for illegal_scale in (0, -1, 2.0, 2.5, float('inf'), float('nan'), True, '2'):
        try:
            fft_hub(POINT, codec_config, mul_config,
                    dict(add_config, scale=[2, 2, illegal_scale]))
        except AssertionError as error:
            assert str(error) == (
                f'Invalid FFT scale <{illegal_scale!r}> at stage <2>; legal '
                f'values are positive Python ints.'
            ), error
        else:
            raise AssertionError(f'fft_hub accepted illegal scale {illegal_scale!r}')

    # Stage 0 runs at twice its requested scale, so its accumulator ceiling is
    # reached at half the requested value the later stages tolerate.
    for stage, scales, effective in ((0, [10, 2, 2], 20), (2, [2, 2, 10], 10)):
        try:
            fft_hub(POINT, codec_config, mul_config,
                    dict(add_config, scale=scales, intwidth=4))
        except AssertionError as error:
            assert str(error) == (
                f'FFT stage <{stage}> construction failed: add_scale scale <{float(effective)}> '
                f'exceeds accumulator maximum <7.0> for intwidth <4> and fracwidth <0>.'
            ), error
        else:
            raise AssertionError(
                f'fft_hub accepted stage-{stage} scale 10 above the intwidth-4 maximum'
            )
    print('Test passed.')


def test_fft_hub_mix_kernel_dims():
    """Verify the mix kernel decorrelates twiddle streams and rejects colliding input dims."""
    codec_config, mul_config, add_config = _configs()
    mix_config = dict(mul_config, kernel='mix')
    operation = fft_hub(POINT, codec_config, mix_config, add_config)
    # Each stage twiddle encoder must hold a sequence neither input encoder uses.
    for stage in range(operation.stages):
        twiddle_encode = getattr(operation.core, f'butterfly_stage_{stage}').reference_encode
        assert not torch.equal(twiddle_encode.num_seq, operation.encode_real.num_seq)
        assert not torch.equal(twiddle_encode.num_seq, operation.encode_imag.num_seq)

    # The three stages take twiddle dimensions 5 to 7, so dim 4 collides through dim + 1.
    for dim in (4, 5, 7):
        try:
            fft_hub(POINT, dict(codec_config, dim=dim), mix_config, add_config)
        except AssertionError as error:
            assert str(error) == (
                f'Invalid dim: <{dim}>; legal values: input encoder dimensions <{dim}> '
                f'and <{dim + 1}> must both stay outside the mix-kernel twiddle '
                f'dimensions <5> through <7>.'
            ), error
        else:
            raise AssertionError(f'fft_hub accepted colliding mix-kernel dim {dim}')

    # Dims just outside the twiddle range accept: dim 3 (dim + 1 = 4 below) and dim 8 above.
    for dim in (3, 8):
        accepted = fft_hub(POINT, dict(codec_config, dim=dim), mix_config, add_config)
        assert accepted.dim == dim

    # The kernel name is matched case-insensitively, so mixed case still rejects dim 5.
    for kernel in ('Mix', 'MIX'):
        try:
            fft_hub(POINT, dict(codec_config, dim=5), dict(mul_config, kernel=kernel), add_config)
        except AssertionError as error:
            assert str(error) == (
                'Invalid dim: <5>; legal values: input encoder dimensions <5> '
                'and <6> must both stay outside the mix-kernel twiddle '
                'dimensions <5> through <7>.'
            ), error
        else:
            raise AssertionError(f'fft_hub accepted colliding dim with kernel {kernel!r}')

    # The guard rejects before the core is built, so a colliding dim leaves no core attribute.
    partial = fft_hub.__new__(fft_hub)
    try:
        fft_hub.__init__(partial, POINT, dict(codec_config, dim=5), mix_config, add_config)
    except AssertionError:
        assert not hasattr(partial, 'core')
    print('Test passed.')


def test_fft_hub_rejects_invalid_shapes():
    """Verify invalid numeric input shapes raise before any child advances."""
    operation = _make_hub('cpu')
    values = torch.zeros(POINT, 1, dtype=global_config.ntype)
    operation.forward_timestep(values, values)
    timesteps = operation.encode_real.timestep_cur
    accumulators = [
        getattr(operation.core, f'butterfly_stage_{stage}').add_y.accumulator.clone()
        for stage in range(operation.stages)
    ]
    for bad, message in (
        ((values, torch.zeros(POINT, 2, dtype=global_config.ntype)),
         'FFT input shapes must match: got <torch.Size([8, 1])> and <torch.Size([8, 2])>.'),
        ((torch.zeros(POINT - 1, 1, dtype=global_config.ntype),) * 2,
         'FFT first input dimension must equal point <8>: got shape <torch.Size([7, 1])>.'),
    ):
        try:
            operation.forward_timestep(*bad)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'fft_hub accepted input shapes {[value.shape for value in bad]}')
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


def test_fft_hub_performance():
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
            lambda values: operation.forward_timestep(*values),
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
    test_fft_hub_rejects_invalid_config()
    test_fft_hub_mix_kernel_dims()
    test_fft_hub_rejects_invalid_shapes()
    test_fft_hub_known_answer()
    test_fft_hub_mixed_scales()
    test_fft_hub_reset_replay()
    test_fft_hub_matches_bare_composition()
    test_fft_hub_matches_timestep_loop()
    test_fft_hub_progressive_precision()
    test_fft_hub_streaming()
    test_fft_hub_performance()
    print('Test passed.')

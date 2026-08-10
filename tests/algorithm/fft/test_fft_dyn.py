"""Test the dynamic-scale spike-port FFT; gradients are exempt because it has no parameters."""

import math

import torch

from napl.sim.algorithm.fft import butterfly_ugemm_dyn, fft_dyn
from napl.sim.base import global_config
from napl.sim.operation import add_any, add_any_dyn, decode, encode
from napl.utils._shared_test import benchmark, devices


POINT = 8
TIMESTEP = 2048
SCALE_MAX = 4
WIDTH = int(math.log2(TIMESTEP)) + 1
# Gate 5's stochastic-computing bound, 1/sqrt(N) at N = 2048 timesteps.
RELATIVE_ERROR_BOUND = 1 / math.sqrt(TIMESTEP)
KNOWN_GAIN_MIN = 0.88
GAIN_MAX = 1.05


def _configs(scale_max=SCALE_MAX, width=WIDTH):
    codec_config = {
        'polarity': 'bipolar',
        'timestep': TIMESTEP,
        'generator': 'sobol',
    }
    mul_config = {
        'polarity': 'bipolar',
        'timestep': TIMESTEP,
        'generator': 'sobol',
    }
    add_config = {'polarity': 'bipolar', 'scale_max': scale_max, 'width': width}
    return codec_config, mul_config, add_config


def _make_fft(device, scale_max=SCALE_MAX, width=WIDTH):
    _, mul_config, add_config = _configs(scale_max, width)
    return fft_dyn(POINT, mul_config, add_config).to(device)


def _stream(operation, values, scales, dims=(1, 2), timesteps=TIMESTEP):
    """Encode the numeric samples, stream them at one scale list, and decode the spectrum."""
    codec_config = _configs()[0]
    device = values[0].device
    encoders = [encode(dict(codec_config, dim=dim)).to(device) for dim in dims]
    decoders = [decode(codec_config).to(device) for _ in range(2)]
    outputs = None
    for _ in range(timesteps):
        outputs = operation(
            *(encoder(value) for encoder, value in zip(encoders, values)), scales
        )
        for decoder, output in zip(decoders, outputs):
            decoder(output)
    decoded = tuple(decoder.spike_value * operation.compensation for decoder in decoders)
    return outputs, decoded


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


def test_fft_dyn_streaming():
    """Verify the bipolar-only dynamic spike-port FFT because subtraction needs signed streams."""
    torch.manual_seed(0)
    full_range = torch.linspace(-1, 1, POINT, dtype=global_config.ntype)
    values_cpu = (
        torch.stack([full_range, full_range.roll(1), full_range.roll(2)], dim=1),
        torch.stack([full_range.flip(0), full_range.roll(3), full_range], dim=1),
    )
    performance_spikes_cpu = (
        torch.randint(0, 2, (POINT, 16384)).type(global_config.stype),
        torch.randint(0, 2, (POINT, 16384)).type(global_config.stype),
    )

    cpu_runtime = None
    for device in devices():
        values = tuple(value.to(device) for value in values_cpu)
        reference = _reference(values)
        for dims in ((1, 2), (3, 4), (5, 6)):
            operation = _make_fft(device)
            assert operation.streaming is True
            assert operation.scale_max == SCALE_MAX
            assert operation.compensation is None
            assert not hasattr(operation, 'scales')
            for stage in range(operation.stages):
                child = getattr(operation, f'butterfly_stage_{stage}')
                assert isinstance(child, butterfly_ugemm_dyn)
                assert isinstance(child.add_y, add_any_dyn)
                assert not isinstance(child.add_y, add_any)
                # This FFT never leaves the spike domain, so no stage owns a codec.
                assert not hasattr(child, 'encoder_x')
                assert not hasattr(child, 'decoder_y')
            assert all(
                isinstance(child, butterfly_ugemm_dyn) for child in operation.children()
            )

            outputs, decoded = _stream(operation, values, [2, 2, 2], dims)
            first = tuple(value.detach().clone() for value in outputs)
            for output in first:
                assert torch.equal(output, output.round())
                assert output.min().item() >= 0 and output.max().item() <= 1
            assert operation.compensation == 2 * 2 * 2 * 2
            assert all(value.shape == values[0].shape for value in first)
            expected_stage_shape = (2 * POINT,) + values[0].shape[1:]
            for stage in range(operation.stages):
                child = getattr(operation, f'butterfly_stage_{stage}')
                assert child.add_y.accumulator.shape == expected_stage_shape
                assert child.timestep_cur == TIMESTEP
                assert child.add_y.timestep_cur == TIMESTEP
            # Stage 0 absorbs the input halving the hub version applies numerically.
            assert operation.butterfly_stage_0.compensation == 4
            assert operation.butterfly_stage_1.compensation == 2

            rmse, relative_bound, gain = _metrics(decoded, reference)
            assert rmse < relative_bound, (
                f'FFT RMSE {rmse:.4f} exceeds relative bound {relative_bound:.4f}'
            )
            assert 0.95 <= gain <= GAIN_MAX, (
                f'FFT gain {gain:.4f} is outside [0.95, {GAIN_MAX}]'
            )
            assert operation.timestep_cur == TIMESTEP
            print(
                f'[{device}][dims={dims}][scales=[2, 2, 2]] rmse={rmse:.4f}, '
                f'relative_bound={relative_bound:.4f}, gain={gain:.4f}'
            )

            if dims == (1, 2):
                operation.reset()
                assert operation.timestep_cur == 0
                assert operation.compensation is None
                for stage in range(operation.stages):
                    child = getattr(operation, f'butterfly_stage_{stage}')
                    assert child.timestep_cur == 0
                    assert child.add_y.timestep_cur == 0
                    assert child.compensation is None
                    assert torch.count_nonzero(child.add_y.accumulator) == 0
                replay, _ = _stream(operation, values, [2, 2, 2], dims)
                assert all(torch.equal(before, after) for before, after in zip(first, replay))

        performance_operation = _make_fft(device)
        device_runtime = benchmark(
            lambda spikes: performance_operation(*spikes, 2),
            performance_spikes_cpu,
            device,
            warmup_runs=1,
            trials=3,
            prepare=performance_operation.reset,
        )
        if device == 'cpu':
            cpu_runtime = device_runtime
        assert cpu_runtime is not None
        print(
            f'[{device}] device_runtime={device_runtime * 1e3:.1f}ms, '
            f'cpu_runtime={cpu_runtime * 1e3:.1f}ms, '
            f'speedup={cpu_runtime / device_runtime:.2f}x'
        )


def test_fft_dyn_known_answers():
    """Verify bipolar impulse and bin-1 tone spectra because subtraction needs signed streams."""
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
            _, decoded = _stream(_make_fft(device), values, 2)
            rmse, relative_bound, gain = _metrics(decoded, expected)
            assert rmse < relative_bound, (
                f'{name} RMSE {rmse:.4f} exceeds relative bound {relative_bound:.4f}'
            )
            assert KNOWN_GAIN_MIN <= gain <= GAIN_MAX, (
                f'{name} gain {gain:.4f} is outside [{KNOWN_GAIN_MIN}, {GAIN_MAX}]'
            )
            print(
                f'[{device}][{name}] rmse={rmse:.4f}, '
                f'relative_bound={relative_bound:.4f}, gain={gain:.4f}'
            )


def test_fft_dyn_mixed_scales():
    """Verify runtime mixed scales preserve the analytic bipolar bin-1 spectrum."""
    runtime_scales = [2, 2, 3]
    for device in devices():
        sample = torch.arange(POINT, dtype=global_config.ntype, device=device)
        angle = 2 * math.pi * sample / POINT
        values = (angle.cos().view(POINT, 1), angle.sin().view(POINT, 1))
        expected_real = torch.zeros_like(values[0])
        expected_real[1].fill_(POINT)
        expected = (expected_real, torch.zeros_like(values[1]))
        operation = _make_fft(device)
        _, decoded = _stream(operation, values, runtime_scales)
        assert operation.compensation == 2 * 2 * 2 * 3
        rmse, relative_bound, gain = _metrics(decoded, expected)
        assert rmse < relative_bound, (
            f'mixed-scale RMSE {rmse:.4f} exceeds relative bound {relative_bound:.4f}'
        )
        assert 0.95 <= gain <= GAIN_MAX, (
            f'mixed-scale gain {gain:.4f} is outside [0.95, {GAIN_MAX}]'
        )
        print(
            f'[{device}][scales={runtime_scales}] rmse={rmse:.4f}, '
            f'relative_bound={relative_bound:.4f}, gain={gain:.4f}'
        )


def test_fft_dyn_scale_change():
    """Verify a runtime scale change affects output while preserving state continuity."""
    for device in devices():
        real = torch.ones(POINT, 1, dtype=global_config.stype, device=device)
        imag = torch.zeros(POINT, 1, dtype=global_config.stype, device=device)
        changed = _make_fft(device)
        held = _make_fft(device)
        for _ in range(8):
            assert all(
                torch.equal(left, right)
                for left, right in zip(changed(real, imag, 2), held(real, imag, 2))
            )
        before = changed.timestep_cur
        changed_output = changed(real, imag, [2, 3, 2])
        held_output = held(real, imag, 2)
        assert changed.timestep_cur == before + 1
        # A scale change alters the carry cadence, so the streams separate within
        # a few timesteps rather than necessarily on the first one.
        diverged = any(
            not torch.equal(left, right)
            for left, right in zip(changed_output, held_output)
        )
        for _ in range(8):
            diverged = diverged or any(
                not torch.equal(left, right)
                for left, right in zip(changed(real, imag, [2, 3, 2]), held(real, imag, 2))
            )
        assert diverged
        assert changed.compensation == 2 * 2 * 3 * 2
        assert all(
            getattr(changed, f'butterfly_stage_{stage}').timestep_cur == changed.timestep_cur
            for stage in range(changed.stages)
        )
        print(f'[{device}] scale_change_timestep={before}->{before + 1}, '
              f'compensation={changed.compensation}')


def test_fft_dyn_rejects_invalid_config():
    """Verify dynamic FFT construction rejects invalid maximum scales by message."""
    _, mul_config, add_config = _configs()
    missing = dict(add_config)
    missing.pop('scale_max')
    try:
        fft_dyn(POINT, mul_config, missing)
    except AssertionError as error:
        assert str(error) == 'Missing key <scale_max> in the dynamic adder configuration.', error
    else:
        raise AssertionError('fft_dyn accepted an add config without scale_max')

    for scale_max in (0, -1, True, 2.0):
        try:
            fft_dyn(POINT, mul_config, dict(add_config, scale_max=scale_max))
        except AssertionError as error:
            assert str(error) == (
                f'fft_dyn scale_max must be a positive int: got <{scale_max}>.'
            ), error
        else:
            raise AssertionError(f'fft_dyn accepted scale_max {scale_max!r}')

    # Stage 0 is built for twice scale_max, so it reaches the accumulator ceiling first.
    try:
        fft_dyn(POINT, mul_config, dict(add_config, scale_max=4, width=4))
    except AssertionError as error:
        assert str(error) == (
            'FFT stage <0> construction failed: add_any scale <8> exceeds '
            'accumulator maximum <7> for width <4>.'
        ), error
    else:
        raise AssertionError('fft_dyn accepted scale_max 4 above the width-4 stage-0 maximum')

    try:
        fft_dyn(3, mul_config, add_config)
    except AssertionError as error:
        assert str(error) == (
            'Invalid point: <3>; legal values: a power-of-two integer '
            'greater than or equal to 2.'
        ), error
    else:
        raise AssertionError('fft_dyn accepted invalid point 3')


def test_fft_dyn_rejects_invalid_scales():
    """Verify invalid runtime controls raise by message before any stage state changes."""
    real = torch.zeros(POINT, 1, dtype=global_config.stype)
    imag = torch.zeros_like(real)
    operation = _make_fft('cpu')
    operation(real, imag, 2)
    parent_timestep = operation.timestep_cur
    accumulators = [
        getattr(operation, f'butterfly_stage_{stage}').add_y.accumulator.clone()
        for stage in range(operation.stages)
    ]
    for bad_scales, message in (
        ([2, 2], 'FFT runtime scale list length <2> must equal the stage count <3>.'),
        ([2, 2, 2, 2], 'FFT runtime scale list length <4> must equal the stage count <3>.'),
        (2.0, 'FFT runtime scales must be a Python int or list: got <2.0>.'),
        (True, 'FFT runtime scales must be a Python int or list: got <True>.'),
        ([2, True, 2], 'FFT runtime scale at stage <1> must be an int: got <True>.'),
        ([2, 0, 2],
         'FFT runtime scale <0> at stage <1> outside the supported range <1> to scale_max <4>.'),
        ([2, 5, 2],
         'FFT runtime scale <5> at stage <1> outside the supported range <1> to scale_max <4>.'),
        ([3, 2, 2],
         'FFT runtime scale <3> at stage <0> must not exceed half of scale_max <4>, '
         'since stage 0 runs at twice its requested scale.'),
        (3, 'FFT runtime scale <3> at stage <0> must not exceed half of scale_max <4>, '
            'since stage 0 runs at twice its requested scale.'),
    ):
        try:
            operation(real, imag, bad_scales)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'fft_dyn accepted runtime scales {bad_scales!r}')
    assert operation.timestep_cur == parent_timestep
    assert all(
        torch.equal(
            getattr(operation, f'butterfly_stage_{stage}').add_y.accumulator,
            accumulators[stage],
        )
        for stage in range(operation.stages)
    )


def test_fft_dyn_rejects_invalid_shapes():
    """Verify invalid input spike shapes raise by message before any stage adder advances."""
    operation = _make_fft('cpu')
    spike = torch.zeros(POINT, 1, dtype=global_config.stype)
    operation(spike, spike, 2)
    accumulators = [
        getattr(operation, f'butterfly_stage_{stage}').add_y.accumulator.clone()
        for stage in range(operation.stages)
    ]
    for bad, message in (
        ((spike, torch.zeros(POINT, 2, dtype=global_config.stype)),
         'FFT input shapes must match: got <torch.Size([8, 1])> and <torch.Size([8, 2])>.'),
        ((torch.zeros(POINT - 1, 1, dtype=global_config.stype),) * 2,
         'FFT first input dimension must equal point <8>: got shape <torch.Size([7, 1])>.'),
    ):
        try:
            operation(*bad, 2)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'fft_dyn accepted input shapes {[value.shape for value in bad]}')
    assert all(
        torch.equal(
            getattr(operation, f'butterfly_stage_{stage}').add_y.accumulator,
            accumulators[stage],
        )
        for stage in range(operation.stages)
    )


if __name__ == '__main__':
    test_fft_dyn_rejects_invalid_config()
    test_fft_dyn_rejects_invalid_scales()
    test_fft_dyn_rejects_invalid_shapes()
    test_fft_dyn_known_answers()
    test_fft_dyn_mixed_scales()
    test_fft_dyn_scale_change()
    test_fft_dyn_streaming()
    print('Test passed.')

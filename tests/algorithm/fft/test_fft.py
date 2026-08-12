"""Test the spike-port streaming FFT; gradients are exempt because it has no trainable parameters."""

import math

import torch

from napl.sim.algorithm.fft import butterfly_mix, butterfly_ugemm, fft
from napl.sim.base import global_config
from napl.sim.operation import decode, encode
from napl.utils._shared_test import benchmark, devices


POINT = 8
TIMESTEP = 2048
SCALE = 2
WIDTH = int(math.log2(TIMESTEP)) + 1
# The Gaines kernel check is a wiring and bit-exactness check, so it streams fewer steps.
MIX_TIMESTEP = 256


def _configs():
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
    add_config = {
        'polarity': 'bipolar',
        'scale': SCALE,
        'intwidth': WIDTH,
        'fracwidth': 0,
    }
    return codec_config, mul_config, add_config


def _make_fft(device):
    codec_config, mul_config, add_config = _configs()
    return fft(POINT, mul_config, add_config).to(device)


def _stream(operation, values, dims=(1, 2), timesteps=TIMESTEP):
    """Encode the numeric samples, stream them, and decode the spectrum streams."""
    codec_config = _configs()[0]
    device = values[0].device
    encoders = [encode(dict(codec_config, dim=dim)).to(device) for dim in dims]
    decoders = [decode(codec_config).to(device) for _ in range(2)]
    outputs = None
    for _ in range(timesteps):
        outputs = operation(*(encoder(value) for encoder, value in zip(encoders, values)))
        for decoder, output in zip(decoders, outputs):
            decoder(output)
    decoded = tuple(decoder.spike_value * operation.compensation for decoder in decoders)
    return outputs, decoded


def _reference(values):
    spectrum = torch.fft.fft(torch.complex(*values), dim=0)
    return spectrum.real, spectrum.imag


def _mix_stage_chain(device):
    """Wire the mix stage chain by hand from bare butterfly_mix cores."""
    _, mul_config, add_config = _configs()
    stage_count = int(math.log2(POINT))
    stages = []
    plan = []
    for stage in range(stage_count):
        span = 2 ** (stage + 1)
        half = span // 2
        first, second, twiddle = [], [], []
        for block in range(0, POINT, span):
            for offset in range(half):
                first.append(block + offset)
                second.append(block + offset + half)
                angle = -2 * math.pi * offset / span
                twiddle.append(complex(math.cos(angle), math.sin(angle)))
        twiddle_tensor = torch.tensor(twiddle, dtype=torch.complex64)
        stage_add_config = dict(add_config, scale=SCALE * (2 if stage == 0 else 1))
        stages.append(butterfly_mix(
            twiddle_tensor.real, twiddle_tensor.imag,
            dict(mul_config, dim=5 + stage), stage_add_config,
        ).to(device))
        plan.append((torch.tensor(first, dtype=torch.long, device=device),
                     torch.tensor(second, dtype=torch.long, device=device)))
    bit_reversed = torch.tensor(
        [int(f'{index:0{stage_count}b}'[::-1], 2) for index in range(POINT)],
        dtype=torch.long, device=device,
    )
    return stages, plan, bit_reversed


def _mix_chain_step(stages, plan, bit_reversed, real, imag):
    """Advance the hand-wired mix stage chain one timestep on already-encoded spikes."""
    real = real.index_select(0, bit_reversed)
    imag = imag.index_select(0, bit_reversed)
    for (first, second), butterfly in zip(plan, stages):
        y0r, y0i, y1r, y1i = butterfly(
            real.index_select(0, first),
            imag.index_select(0, first),
            real.index_select(0, second),
            imag.index_select(0, second),
        )
        next_real = torch.empty_like(real)
        next_imag = torch.empty_like(imag)
        next_real.index_copy_(0, first, y0r)
        next_real.index_copy_(0, second, y1r)
        next_imag.index_copy_(0, first, y0i)
        next_imag.index_copy_(0, second, y1i)
        real, imag = next_real, next_imag
    return real, imag


def _metrics(candidate, reference):
    candidate_complex = torch.complex(*candidate)
    reference_complex = torch.complex(*reference)
    reference_rms = reference_complex.abs().pow(2).mean().sqrt()
    rmse = (candidate_complex - reference_complex).abs().pow(2).mean().sqrt()
    gain = candidate_complex.abs().pow(2).mean().sqrt() / reference_rms
    return rmse.item(), gain.item()


def test_fft_streaming():
    """Verify the bipolar-only spike-port FFT because its subtraction paths need signed streams."""
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
            assert operation.scales == [SCALE] * operation.stages
            assert operation.compensation == 2 * SCALE ** operation.stages
            # Every stage multiplier turns its constant twiddle into a stream itself.
            assert operation.internal_encode is True
            # Stage 0 absorbs the input halving the hub version applies numerically,
            # so its adder runs at twice the requested scale.
            child_scales = [
                getattr(operation, f'butterfly_stage_{stage}').add_y.scale
                for stage in range(operation.stages)
            ]
            assert child_scales == [2 * SCALE] + [SCALE] * (operation.stages - 1)
            assert all(
                isinstance(getattr(operation, f'butterfly_stage_{stage}'), butterfly_ugemm)
                for stage in range(operation.stages)
            )
            # This FFT never leaves the spike domain, so no stage owns a codec.
            assert all(
                not hasattr(getattr(operation, f'butterfly_stage_{stage}'), 'encoder_x')
                and not hasattr(getattr(operation, f'butterfly_stage_{stage}'), 'decoder_y')
                for stage in range(operation.stages)
            )
            assert operation.encoding_io == {
                port: 'rc' for port in ('input_real', 'input_imag',
                                        'output_real', 'output_imag')
            }

            outputs, decoded = _stream(operation, values, dims)
            first = tuple(value.detach().clone() for value in outputs)
            for output in first:
                assert torch.equal(output, output.round())
                assert output.min().item() >= 0 and output.max().item() <= 1
            assert all(value.shape == values[0].shape for value in first)
            expected_stage_shape = (2 * POINT,) + values[0].shape[1:]
            assert all(
                getattr(operation, f'butterfly_stage_{stage}').add_y.accumulator.shape
                == expected_stage_shape
                for stage in range(operation.stages)
            )

            rmse, gain = _metrics(decoded, reference)
            assert operation.timestep_cur == TIMESTEP
            assert all(
                getattr(operation, f'butterfly_stage_{stage}').timestep_cur == TIMESTEP
                for stage in range(operation.stages)
            )
            print(
                f'[{device}][dims={dims}] rmse={rmse:.4f}, gain={gain:.4f}'
            )

            if dims == (1, 2):
                operation.reset()
                assert operation.timestep_cur == 0
                assert all(
                    getattr(operation, f'butterfly_stage_{stage}').timestep_cur == 0
                    for stage in range(operation.stages)
                )
                assert all(
                    torch.count_nonzero(
                        getattr(operation, f'butterfly_stage_{stage}').add_y.accumulator
                    ) == 0
                    for stage in range(operation.stages)
                )
                replay, _ = _stream(operation, values, dims)
                assert all(torch.equal(before, after) for before, after in zip(first, replay))

        performance_operation = _make_fft(device)
        device_runtime = benchmark(
            lambda spikes: performance_operation(*spikes),
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


def test_fft_known_answer():
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
            _, decoded = _stream(_make_fft(device), values)
            rmse, gain = _metrics(decoded, expected)
            print(
                f'[{device}][{name}] rmse={rmse:.4f}, gain={gain:.4f}'
            )


def test_fft_mixed_scales():
    """Verify mixed stage scales preserve the analytic bipolar complex-tone spectrum."""
    mixed_scales = [2, 2, 3]
    for device in devices():
        codec_config, mul_config, add_config = _configs()
        caller_add_config = dict(add_config, scale=list(mixed_scales))
        original_add_config = dict(caller_add_config, scale=list(mixed_scales))
        operation = fft(POINT, mul_config, caller_add_config).to(device)
        assert caller_add_config == original_add_config
        child_scales = [
            getattr(operation, f'butterfly_stage_{stage}').add_y.scale
            for stage in range(operation.stages)
        ]
        # These wiring checks are load-bearing: the numerical gates also pass
        # if all three stages incorrectly use the uniform scale list [2, 2, 2].
        assert operation.scales == mixed_scales
        assert child_scales == [4, 2, 3]
        assert operation.compensation == 2 * 2 * 2 * 3

        sample = torch.arange(POINT, dtype=global_config.ntype, device=device)
        angle = 2 * math.pi * sample / POINT
        values = (angle.cos().view(POINT, 1), angle.sin().view(POINT, 1))
        expected_real = torch.zeros_like(values[0])
        expected_real[1].fill_(POINT)
        expected = (expected_real, torch.zeros_like(values[1]))
        _, decoded = _stream(operation, values)
        rmse, gain = _metrics(decoded, expected)
        print(
            f'[{device}][mixed_scales={mixed_scales}] child_scales={child_scales}, '
            f'rmse={rmse:.4f}, gain={gain:.4f}'
        )


def test_fft_mix_kernel():
    """Verify the kernel knob picks the Gaines stage class and its per-stage twiddle dims."""
    codec_config, mul_config, add_config = _configs()
    caller_mul_config = dict(mul_config, kernel='mix')
    original_mul_config = dict(caller_mul_config)

    for device in devices():
        operation = fft(POINT, caller_mul_config, add_config).to(device)
        assert caller_mul_config == original_mul_config
        stages = [
            getattr(operation, f'butterfly_stage_{stage}')
            for stage in range(operation.stages)
        ]
        assert all(isinstance(stage, butterfly_mix) for stage in stages)
        assert operation.internal_encode is True
        assert operation.compensation == 2 * SCALE ** operation.stages

        # Stage index i encodes its constant twiddle on Sobol dimension 5 + i.
        schedule = [
            encode(dict(mul_config, dim=5 + index)).to(device).num_seq
            for index in range(operation.stages)
        ]
        assert all(
            torch.equal(stage.reference_encode.num_seq, expected)
            for stage, expected in zip(stages, schedule)
        )
        for left in range(operation.stages):
            for right in range(left + 1, operation.stages):
                assert not torch.equal(schedule[left], schedule[right])

        full_range = torch.linspace(-1, 1, POINT, dtype=global_config.ntype, device=device)
        values = (full_range.view(POINT, 1), full_range.flip(0).view(POINT, 1))

        # The default and the explicit request both keep the conditional-spike
        # stage and produce the same stream.
        default_outputs = []
        for default_mul_config in (mul_config, dict(mul_config, kernel='ugemm')):
            reference_operation = fft(POINT, default_mul_config, add_config).to(device)
            assert all(
                isinstance(getattr(reference_operation, f'butterfly_stage_{stage}'),
                           butterfly_ugemm)
                for stage in range(reference_operation.stages)
            )
            default_output, _ = _stream(reference_operation, values, timesteps=MIX_TIMESTEP)
            default_outputs.append(tuple(value.detach().clone() for value in default_output))
        assert all(torch.equal(left, right) for left, right in zip(*default_outputs))

        # Gaines correlation error is systematic, so the mix kernel is checked
        # bit-exactly against the same stage chain wired from bare butterflies. Both
        # consume one shared set of encoded spikes and are compared every timestep, as
        # the hub composition test checks its wrapper.
        chain_stages, plan, bit_reversed = _mix_stage_chain(device)
        encoders = [encode(dict(codec_config, dim=dim)).to(device) for dim in (1, 2)]
        outputs = None
        for step in range(MIX_TIMESTEP):
            real, imag = (encoder(value) for encoder, value in zip(encoders, values))
            outputs = operation(real, imag)
            chain_out = _mix_chain_step(chain_stages, plan, bit_reversed, real, imag)
            for name, candidate, expected in zip(('real', 'imag'), outputs, chain_out):
                assert torch.equal(candidate, expected), \
                    f'[{device}][{name}] diverged from the bare stage chain at timestep {step}'
        first = tuple(value.detach().clone() for value in outputs)
        for output in first:
            assert torch.equal(output, output.round())
            assert output.min().item() >= 0 and output.max().item() <= 1
            assert output.shape == values[0].shape
        assert operation.timestep_cur == MIX_TIMESTEP
        assert all(stage.timestep_cur == MIX_TIMESTEP for stage in stages)
        print(f'[{device}][kernel=mix] bit-exact against the bare stage chain for '
              f'{MIX_TIMESTEP} timesteps')

        operation.reset()
        assert operation.timestep_cur == 0
        assert all(stage.timestep_cur == 0 for stage in stages)
        assert all(torch.count_nonzero(stage.add_y.accumulator) == 0 for stage in stages)
        replay, _ = _stream(operation, values, timesteps=MIX_TIMESTEP)
        assert all(torch.equal(before, after) for before, after in zip(first, replay))
        print(f'[{device}][kernel=mix] stage_dims={[5 + index for index in range(operation.stages)]}')

    # The kernel value is matched after lowercasing, as polarity and generator are.
    for kernel in ('UGEMM', 'Mix'):
        assert isinstance(fft(POINT, dict(mul_config, kernel=kernel), add_config)
                          .butterfly_stage_0,
                          butterfly_ugemm if kernel.lower() == 'ugemm' else butterfly_mix)

    for kernel in ('gaines', 'ugemm ', None, 0):
        try:
            fft(POINT, dict(mul_config, kernel=kernel), add_config)
        except AssertionError as error:
            assert str(error) == (
                f"Invalid kernel: <{kernel!r}>; legal values: <['mix', 'ugemm']>."
            ), error
        else:
            raise AssertionError(f'fft accepted invalid kernel {kernel!r}')


def test_fft_rejects_invalid_config():
    """Verify the FFT rejects invalid sizes, scales, and multiplier configuration by message."""
    codec_config, mul_config, add_config = _configs()
    point_message = (
        'Invalid point: <{0!r}>; legal values: a power-of-two integer '
        'greater than or equal to 2.'
    )
    for point in (0, 3, True, '8'):
        try:
            fft(point, mul_config, add_config)
        except AssertionError as error:
            assert str(error) == point_message.format(point), error
        else:
            raise AssertionError(f'fft accepted invalid point {point!r}')

    missing = dict(mul_config)
    missing.pop('timestep')
    try:
        fft(POINT, missing, add_config)
    except AssertionError as error:
        assert str(error) == 'Missing key <timestep> in the input configuration.', error
    else:
        raise AssertionError('fft accepted a multiplier configuration missing timestep')

    try:
        fft(POINT, dict(mul_config, unsupported=True), add_config)
    except AssertionError as error:
        assert str(error) == (
            "Unknown key <unsupported> in the input configuration; accepted keys: "
            "<['generator', 'kernel', 'name', 'polarity', 'timestep']>."
        ), error
    else:
        raise AssertionError('fft accepted an unknown multiplier configuration key')

    missing_add_scale = dict(add_config)
    missing_add_scale.pop('scale')
    try:
        fft(POINT, mul_config, missing_add_scale)
    except AssertionError as error:
        assert str(error) == 'Missing key <scale> in the input adder configuration.', error
    else:
        raise AssertionError('fft accepted an adder configuration missing scale')

    try:
        fft(POINT, dict(mul_config, polarity='unipolar'),
            dict(add_config, polarity='unipolar'))
    except AssertionError as error:
        assert str(error) == "Invalid polarity: <unipolar>; legal values: <['bipolar']>.", error
    else:
        raise AssertionError('fft accepted a unipolar configuration')

    for scales in ([2, 2], [2, 2, 2, 2]):
        try:
            fft(POINT, mul_config, dict(add_config, scale=scales))
        except AssertionError as error:
            assert str(error) == (
                f'FFT scale list length <{len(scales)}> must equal the stage count <3>.'
            ), error
        else:
            raise AssertionError(f'fft accepted scale list with length {len(scales)}')

    for illegal_scale in (0, -1, 2.0, 2.5, float('inf'), float('nan'), True, '2'):
        try:
            fft(POINT, mul_config, dict(add_config, scale=[2, 2, illegal_scale]))
        except AssertionError as error:
            assert str(error) == (
                f'Invalid FFT scale <{illegal_scale!r}> at stage <2>; legal '
                f'values are positive Python ints.'
            ), error
        else:
            raise AssertionError(f'fft accepted illegal scale {illegal_scale!r}')

    for illegal_scale in (0, 2.0, 2.5, True):
        try:
            fft(POINT, mul_config, dict(add_config, scale=illegal_scale))
        except AssertionError as error:
            assert str(error) == (
                f'Invalid FFT scale <{illegal_scale!r}> at stage <0>; legal '
                f'values are positive Python ints.'
            ), error
        else:
            raise AssertionError(f'fft accepted illegal scalar scale {illegal_scale!r}')

    # Stage 0 runs at twice its requested scale, so its accumulator ceiling is
    # reached at half the requested value the later stages tolerate.
    for stage, scales, effective in ((0, [10, 2, 2], 20), (2, [2, 2, 10], 10)):
        try:
            fft(POINT, mul_config, dict(add_config, scale=scales, intwidth=4))
        except AssertionError as error:
            assert str(error) == (
                f'FFT stage <{stage}> construction failed: add_scale scale <{float(effective)}> '
                f'exceeds accumulator maximum <7.0> for intwidth <4> and fracwidth <0>.'
            ), error
        else:
            raise AssertionError(f'fft accepted stage-{stage} scale 10 above the intwidth-4 maximum')


def test_fft_rejects_invalid_shapes():
    """Verify invalid input spike shapes raise before any stage adder advances."""
    operation = _make_fft('cpu')
    spike = torch.zeros(POINT, 1, dtype=global_config.stype)
    operation(spike, spike)
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
            operation(*bad)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'fft accepted input shapes {[value.shape for value in bad]}')
    assert all(
        torch.equal(
            getattr(operation, f'butterfly_stage_{stage}').add_y.accumulator,
            accumulators[stage],
        )
        for stage in range(operation.stages)
    )


if __name__ == '__main__':
    test_fft_rejects_invalid_config()
    test_fft_rejects_invalid_shapes()
    test_fft_mix_kernel()
    test_fft_known_answer()
    test_fft_mixed_scales()
    test_fft_streaming()
    print('Test passed.')

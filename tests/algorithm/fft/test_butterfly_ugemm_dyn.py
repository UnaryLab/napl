"""Test the dynamic-scale spike-port butterfly; gradients are exempt because it has no parameters."""

import math

import torch

from napl.sim.algorithm.fft import butterfly_fp, butterfly_ugemm, butterfly_ugemm_dyn
from napl.sim.base import global_config
from napl.sim.operation import add_any, add_any_dyn, decode, encode
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices


LANE = 512
TIMESTEP = 256
WIDTH = int(math.log2(TIMESTEP))
SCALE_MAX = 4
SCALE = 3
# Gate 5's 1/sqrt(N) stochastic-computing form with a measured constant: 0.4/sqrt(256) =
# 0.025000. Standing constraint: the constant is measured over the legal bipolar input
# range, and it is valid only because the Sobol streams make the run deterministic for the
# seed, inputs, twiddles, lane count, and timestep below. If any of those change, this
# constant must be RE-MEASURED across the legal input range, not hand-adjusted until a run
# passes. Measured worst clean cases: 0.016862 over random per-lane inputs (seeds 0-7, with
# and without a +-0.3 shift) and 0.022285 over constant-valued inputs, so the bound clears
# the random family by 1.48x and the constant family by 1.12x. It catches a twiddle sign
# flip (0.956078) and a 5% twiddle scale error (0.026282, only 1.05x over the bound). It
# does not catch a 2% single-stage twiddle scale error (0.018273): the clean spread across
# legal inputs is wider than that signal, so no threshold on this check separates the two.
RMSE_BOUND = 0.4 / math.sqrt(TIMESTEP)


def _configs(scale_max=SCALE_MAX, width=WIDTH + 1):
    codec_config = {
        'polarity': 'bipolar',
        'timestep': TIMESTEP,
        'generator': 'sobol',
    }
    add_config = {
        'polarity': 'bipolar',
        'scale_max': scale_max,
        'width': width,
    }
    return codec_config, add_config


def _stream(operation, values, codec_config, scale, timesteps=TIMESTEP):
    """Encode the four numeric operands, stream them at one scale, and decode the outputs."""
    encoders = [encode(dict(codec_config, dim=dim)).to(values[0].device) for dim in (1, 2, 3, 4)]
    decoders = [decode(codec_config).to(values[0].device) for _ in range(4)]
    outputs = None
    for _ in range(timesteps):
        outputs = operation(
            *(encoder(value) for encoder, value in zip(encoders, values)), scale
        )
        for decoder, output in zip(decoders, outputs):
            decoder(output)
    decoded = tuple(decoder.spike_value * operation.compensation for decoder in decoders)
    return outputs, decoded


def test_butterfly_ugemm_dyn_streaming():
    """Verify the bipolar-only dynamic spike-port butterfly matches butterfly_fp per device."""
    torch.manual_seed(0)
    codec_config, add_config = _configs()
    values_cpu = tuple(
        gen_rand_tensor('bipolar', shape=(LANE, 1), width=WIDTH).type(global_config.ntype)
        for _ in range(4)
    )
    twiddle_real_cpu = gen_rand_tensor(
        'bipolar', shape=(LANE,), width=WIDTH
    ).type(global_config.ntype)
    twiddle_imag_cpu = gen_rand_tensor(
        'bipolar', shape=(LANE,), width=WIDTH
    ).type(global_config.ntype)
    performance_spikes_cpu = tuple(
        torch.randint(0, 2, (LANE, 256)).type(global_config.stype) for _ in range(4)
    )

    cpu_runtime = None
    for device in devices():
        values = tuple(value.to(device) for value in values_cpu)
        twiddle_real = twiddle_real_cpu.to(device)
        twiddle_imag = twiddle_imag_cpu.to(device)
        operation = butterfly_ugemm_dyn(
            twiddle_real, twiddle_imag, codec_config, add_config
        ).to(device)
        assert isinstance(operation, butterfly_ugemm)
        assert isinstance(operation.add_y, add_any_dyn)
        assert not isinstance(operation.add_y, add_any)
        assert operation.scale_max == SCALE_MAX
        assert operation.compensation is None
        assert not hasattr(operation, 'encoder_x')
        assert not hasattr(operation, 'decoder_y')

        outputs, decoded = _stream(operation, values, codec_config, SCALE)
        first = tuple(value.detach().clone() for value in outputs)
        for output in first:
            assert torch.equal(output, output.round())
            assert output.min().item() >= 0 and output.max().item() <= 1
        assert operation.compensation == SCALE
        assert all(value.shape == values[0].shape for value in first)
        assert operation.add_y.accumulator.shape == (4 * LANE,) + values[0].shape[1:]

        reference = butterfly_fp().to(device)(
            *values, twiddle_real.view(LANE, 1), twiddle_imag.view(LANE, 1)
        )
        for name, candidate, expected in zip(
            ('y0r', 'y0i', 'y1r', 'y1i'), decoded, reference
        ):
            rmse = (candidate - expected).pow(2).mean().sqrt().item()
            # mul_ugemm compares a |w|-derived probability against its number sequence, so a
            # twiddle already at the clamp boundary (|w| = 1) saturates and a magnitude
            # increase on it is invisible here: perturbing only those entries leaves this
            # rmse bit-identical to the clean run. These twiddles hold 4 real and 3 imag
            # entries at the boundary out of 512 lanes; the rest sit below it, which is why
            # a x1.02 magnitude perturbation moves this rmse at all.
            assert rmse < RMSE_BOUND, f'{name} RMSE {rmse:.6f} exceeds bound {RMSE_BOUND:.6f}'
            print(f'[{device}][{name}] rmse={rmse:.6f}, bound={RMSE_BOUND:.6f}')
        assert operation.timestep_cur == TIMESTEP
        assert operation.add_y.timestep_cur == TIMESTEP

        operation.reset()
        assert operation.timestep_cur == 0
        assert operation.compensation is None
        assert operation.add_y.timestep_cur == 0
        assert torch.count_nonzero(operation.add_y.accumulator) == 0
        replay, _ = _stream(operation, values, codec_config, SCALE)
        assert all(torch.equal(before, after) for before, after in zip(first, replay))

        performance_operation = butterfly_ugemm_dyn(
            twiddle_real, twiddle_imag, codec_config, add_config
        ).to(device)
        device_runtime = benchmark(
            lambda spikes: performance_operation(*spikes, SCALE),
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


def test_butterfly_ugemm_dyn_known_answer():
    """Verify a zero twiddle passes the first input spike through at every runtime scale."""
    codec_config, add_config = _configs()
    for device in devices():
        zeros = torch.zeros(4, 1, dtype=global_config.ntype, device=device)
        x0r = torch.full((4, 1), 0.5, dtype=global_config.ntype, device=device)
        twiddle = torch.zeros(4, dtype=global_config.ntype, device=device)
        for scale in (2, 3, 4):
            operation = butterfly_ugemm_dyn(
                twiddle, twiddle, codec_config, add_config
            ).to(device)
            _, decoded = _stream(operation, (x0r, zeros, zeros, zeros), codec_config, scale)
            # With w = 0 and x1 = 0 the twiddle term vanishes: y0 = y1 = x0.
            assert operation.compensation == scale
            # Half the decoded output step at this runtime scale, which is
            # scale * 2 / TIMESTEP wide. The observed residue is 0.007812 at scale 3 and 0
            # at scales 2 and 4, so the scale-3 case holds a 1.5x margin and still catches a
            # 2% offset on x0r, which lands at 0.015625.
            atol = scale / TIMESTEP
            for name, value in (('y0r', decoded[0]), ('y1r', decoded[2])):
                torch.testing.assert_close(value, x0r, atol=atol, rtol=0)
                print(f'[{device}][scale={scale}] {name} '
                      f'max_err={(value - x0r).abs().max().item():.4f}, atol={atol:.6f}')
            for value in (decoded[1], decoded[3]):
                torch.testing.assert_close(value, zeros, atol=atol, rtol=0)


def test_butterfly_ugemm_dyn_scale_change():
    """Verify a runtime scale change alters the output while state advances exactly once."""
    codec_config, add_config = _configs()
    for device in devices():
        twiddle = torch.full((4,), 0.5, dtype=global_config.ntype, device=device)
        spike = torch.ones(4, 1, dtype=global_config.stype, device=device)
        changed = butterfly_ugemm_dyn(
            twiddle, twiddle, codec_config, add_config
        ).to(device)
        held = butterfly_ugemm_dyn(
            twiddle, twiddle, codec_config, add_config
        ).to(device)
        for _ in range(8):
            assert all(
                torch.equal(left, right)
                for left, right in zip(changed(spike, spike, spike, spike, 2),
                                       held(spike, spike, spike, spike, 2))
            )
        before = changed.timestep_cur
        changed_output = changed(spike, spike, spike, spike, 1)
        held_output = held(spike, spike, spike, spike, 2)
        assert any(
            not torch.equal(left, right)
            for left, right in zip(changed_output, held_output)
        )
        assert changed.timestep_cur == before + 1
        assert changed.add_y.timestep_cur == before + 1
        assert changed.compensation == 1
        print(f'[{device}] scale_change_timestep={before}->{before + 1}')


def test_butterfly_ugemm_dyn_rejects_invalid_config():
    """Verify construction and runtime scales are rejected with their exact messages."""
    codec_config, add_config = _configs()
    twiddle = torch.zeros(4, dtype=global_config.ntype)

    missing = dict(add_config)
    missing.pop('scale_max')
    try:
        butterfly_ugemm_dyn(twiddle, twiddle, codec_config, missing)
    except AssertionError as error:
        assert str(error) == 'Missing key <scale_max> in the dynamic adder configuration.', error
    else:
        raise AssertionError('butterfly_ugemm_dyn accepted an add config without scale_max')

    for scale_max in (0, -1, True, 2.0):
        try:
            butterfly_ugemm_dyn(
                twiddle, twiddle, codec_config, dict(add_config, scale_max=scale_max)
            )
        except AssertionError as error:
            assert str(error) == (
                f'butterfly_ugemm_dyn scale_max must be a positive int: got <{scale_max}>.'
            ), error
        else:
            raise AssertionError(f'butterfly_ugemm_dyn accepted scale_max {scale_max!r}')

    try:
        butterfly_ugemm_dyn(
            twiddle, twiddle, dict(codec_config, polarity='unipolar'),
            dict(add_config, polarity='unipolar')
        )
    except AssertionError as error:
        assert str(error) == "Invalid polarity: <unipolar>; legal values: <['bipolar']>.", error
    else:
        raise AssertionError('butterfly_ugemm_dyn accepted a unipolar configuration')

    operation = butterfly_ugemm_dyn(twiddle, twiddle, codec_config, add_config)
    spike = torch.zeros(4, 1, dtype=global_config.stype)
    operation(spike, spike, spike, spike, 2)
    timestep = operation.timestep_cur
    accumulator = operation.add_y.accumulator.clone()
    for scale, message in (
        (2.0, 'butterfly_ugemm_dyn scale must be an int: got <2.0>.'),
        (True, 'butterfly_ugemm_dyn scale must be an int: got <True>.'),
        ('2', "butterfly_ugemm_dyn scale must be an int: got <2>."),
        (0, 'butterfly_ugemm_dyn scale <0> outside the supported range <1> to scale_max <4>.'),
        (5, 'butterfly_ugemm_dyn scale <5> outside the supported range <1> to scale_max <4>.'),
    ):
        try:
            operation(spike, spike, spike, spike, scale)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'butterfly_ugemm_dyn accepted runtime scale {scale!r}')
    # An invalid scale raises inside __call__, before the timestep or adder advance.
    assert operation.timestep_cur == timestep
    assert torch.equal(operation.add_y.accumulator, accumulator)


if __name__ == '__main__':
    test_butterfly_ugemm_dyn_rejects_invalid_config()
    test_butterfly_ugemm_dyn_known_answer()
    test_butterfly_ugemm_dyn_scale_change()
    test_butterfly_ugemm_dyn_streaming()
    print('Test passed.')

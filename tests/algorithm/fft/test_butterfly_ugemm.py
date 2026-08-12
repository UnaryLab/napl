"""Test the spike-port butterfly; gradients are exempt because it has no trainable parameters."""

import math

import torch

from napl.sim.algorithm.fft import butterfly_fp, butterfly_ugemm
from napl.sim.base import global_config
from napl.sim.operation import decode, encode
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices


LANE = 512
TIMESTEP = 256
WIDTH = int(math.log2(TIMESTEP))
SCALE = 3


def _configs():
    codec_config = {
        'polarity': 'bipolar',
        'timestep': TIMESTEP,
        'generator': 'sobol',
    }
    add_config = {
        'polarity': 'bipolar',
        'scale': SCALE,
        'intwidth': WIDTH + 1,
        'fracwidth': 0,
    }
    return codec_config, add_config


def _stream(operation, values, codec_config, timesteps=TIMESTEP):
    """Encode the four numeric operands, stream them, and decode the four outputs."""
    encoders = [encode(dict(codec_config, dim=dim)).to(values[0].device) for dim in (1, 2, 3, 4)]
    decoders = [decode(codec_config).to(values[0].device) for _ in range(4)]
    outputs = None
    for _ in range(timesteps):
        outputs = operation(*(encoder(value) for encoder, value in zip(encoders, values)))
        for decoder, output in zip(decoders, outputs):
            decoder(output)
    decoded = tuple(decoder.spike_value * operation.compensation for decoder in decoders)
    return outputs, decoded


def _assert_spikes(outputs):
    for output in outputs:
        assert torch.equal(output, output.round())
        assert output.min().item() >= 0 and output.max().item() <= 1


def test_butterfly_ugemm_streaming():
    """Verify the bipolar-only spike-port butterfly matches butterfly_fp across devices."""
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
        operation = butterfly_ugemm(
            twiddle_real, twiddle_imag, codec_config, add_config
        ).to(device)
        assert operation.streaming is True
        assert operation.lane == LANE
        assert operation.compensation == SCALE
        # The multiplier turns the constant twiddle into a stream itself.
        assert operation.internal_encode is True
        # The spike ports are the whole point of this class, so its interface
        # must declare rate coding rather than the hub family's empty mapping.
        assert operation.encoding_io == {port: 'rc' for port in
                                         ('x0r', 'x0i', 'x1r', 'x1i', 'y0r', 'y0i', 'y1r', 'y1i')}
        assert not hasattr(operation, 'encoder_x')
        assert not hasattr(operation, 'decoder_y')

        outputs, decoded = _stream(operation, values, codec_config)
        first = tuple(value.detach().clone() for value in outputs)
        _assert_spikes(first)
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
            print(f'[{device}][{name}] rmse={rmse:.6f}')
        assert operation.timestep_cur == TIMESTEP
        assert operation.mul_wx.timestep_cur == TIMESTEP
        assert operation.add_y.timestep_cur == TIMESTEP

        operation.reset()
        assert operation.timestep_cur == 0
        assert operation.mul_wx.timestep_cur == 0
        assert operation.add_y.timestep_cur == 0
        assert torch.count_nonzero(operation.add_y.accumulator) == 0
        assert torch.count_nonzero(operation.mul_wx.seq_idx) == 0
        replay, _ = _stream(operation, values, codec_config)
        assert all(torch.equal(before, after) for before, after in zip(first, replay))

        performance_operation = butterfly_ugemm(
            twiddle_real, twiddle_imag, codec_config, add_config
        ).to(device)
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


def test_butterfly_ugemm_known_answer():
    """Verify a zero twiddle passes the first input spike through at the adder scale."""
    codec_config, add_config = _configs()
    for device in devices():
        zeros = torch.zeros(4, 1, dtype=global_config.ntype, device=device)
        x0r = torch.full((4, 1), 0.5, dtype=global_config.ntype, device=device)
        twiddle = torch.zeros(4, dtype=global_config.ntype, device=device)
        operation = butterfly_ugemm(
            twiddle, twiddle, codec_config, add_config
        ).to(device)
        _, decoded = _stream(operation, (x0r, zeros, zeros, zeros), codec_config)
        # With w = 0 and x1 = 0 the twiddle term vanishes: y0 = y1 = x0.
        for name, value in (('y0r', decoded[0]), ('y1r', decoded[2])):
            print(f'[{device}] {name} max_err={(value - x0r).abs().max().item():.4f}')
        for name, value in (('y0i', decoded[1]), ('y1i', decoded[3])):
            print(f'[{device}] {name} max_err={(value - zeros).abs().max().item():.4f}')


def test_butterfly_ugemm_rejects_invalid_config():
    """Verify construction and calls reject unipolar, bad twiddles, and mismatched shapes."""
    codec_config, add_config = _configs()
    twiddle = torch.zeros(4, dtype=global_config.ntype)

    unipolar = dict(codec_config, polarity='unipolar')
    try:
        butterfly_ugemm(twiddle, twiddle, unipolar, dict(add_config, polarity='unipolar'))
    except AssertionError as error:
        assert str(error) == "Invalid polarity: <unipolar>; legal values: <['bipolar']>.", error
    else:
        raise AssertionError('butterfly_ugemm accepted a unipolar configuration')

    for label, bad_real, bad_imag in (
        ('twiddle_real', torch.zeros(2, 2, dtype=global_config.ntype), twiddle),
        ('twiddle_real', torch.zeros(0, dtype=global_config.ntype), twiddle),
        ('twiddle_imag', twiddle, torch.zeros(4, 1, dtype=global_config.ntype)),
    ):
        value = bad_real if label == 'twiddle_real' else bad_imag
        expected = (
            f'Invalid {label}: <{tuple(value.shape)}>; legal values: a non-empty 1-D tensor.'
        )
        try:
            butterfly_ugemm(bad_real, bad_imag, codec_config, add_config)
        except AssertionError as error:
            assert str(error) == expected, error
        else:
            raise AssertionError(f'butterfly_ugemm accepted {label} shape {tuple(value.shape)}')

    try:
        butterfly_ugemm('twiddle', twiddle, codec_config, add_config)
    except AssertionError as error:
        assert str(error) == (
            'Invalid twiddle_real: <str>; legal values: a non-empty 1-D tensor.'
        ), error
    else:
        raise AssertionError('butterfly_ugemm accepted a non-tensor twiddle_real')

    try:
        butterfly_ugemm(twiddle, torch.zeros(2, dtype=global_config.ntype),
                        codec_config, add_config)
    except AssertionError as error:
        assert str(error) == (
            'butterfly_ugemm twiddle_real length <4> must equal twiddle_imag length <2>.'
        ), error
    else:
        raise AssertionError('butterfly_ugemm accepted mismatched twiddle lengths')

    try:
        butterfly_ugemm(twiddle, twiddle, codec_config,
                        dict(add_config, scale=10, intwidth=4))
    except AssertionError as error:
        assert str(error) == (
            'add_scale scale <10.0> exceeds accumulator maximum <7.0> for intwidth <4> '
            'and fracwidth <0>.'
        ), error
    else:
        raise AssertionError('butterfly_ugemm accepted scale 10 above the width-4 maximum')

    try:
        butterfly_ugemm(twiddle, twiddle, codec_config,
                        dict(add_config, scale=6, intwidth=4))
    except AssertionError as error:
        assert str(error) == (
            'butterfly_ugemm accumulator maximum <7> raw units too small for fan-in <3> and '
            'scale <6.0>: for this bipolar-only adder acc_max must be >= (scale_raw - '
            'grid) + delta_max, with grid <0.5> and delta_max <4.5> in raw units of '
            '<1.0>, and acc_max + 1 must be > entry when scale < entry, or partial sums '
            'saturate. Increase intwidth.'
        ), error
    else:
        raise AssertionError('butterfly_ugemm accepted an accumulator width below the bound')

    operation = butterfly_ugemm(twiddle, twiddle, codec_config, add_config)
    spike = torch.zeros(4, 1, dtype=global_config.stype)
    operation(spike, spike, spike, spike)
    timestep = operation.timestep_cur
    accumulator = operation.add_y.accumulator.clone()
    for bad, message in (
        ((spike, torch.zeros(4, 2, dtype=global_config.stype), spike, spike),
         'butterfly_ugemm input shapes must match: got <torch.Size([4, 1])>, '
         '<torch.Size([4, 2])>, <torch.Size([4, 1])>, and <torch.Size([4, 1])>.'),
        ((torch.zeros(3, 1, dtype=global_config.stype),) * 4,
         'butterfly_ugemm first input dimension must equal lane <4>: '
         'got shape <torch.Size([3, 1])>.'),
    ):
        try:
            operation(*bad)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'butterfly_ugemm accepted input shapes {[b.shape for b in bad]}')
    # The rejection happens after __call__ ticks, so only the adder must stand still.
    assert operation.timestep_cur == timestep + 2
    assert torch.equal(operation.add_y.accumulator, accumulator)


if __name__ == '__main__':
    test_butterfly_ugemm_rejects_invalid_config()
    test_butterfly_ugemm_known_answer()
    test_butterfly_ugemm_streaming()
    print('Test passed.')

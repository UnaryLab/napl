"""Test the Gaines spike-port butterfly; gradients are exempt because it has no trainable parameters."""

import math

import torch

from napl.sim.algorithm.fft import butterfly_fp, butterfly_mix
from napl.sim.base import global_config
from napl.sim.operation import add_scale, decode, encode, mul_gaines
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices


LANE = 512
TIMESTEP = 256
WIDTH = int(math.log2(TIMESTEP))
SCALE = 3
TWIDDLE_DIM = 5


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


def _bare_core_stream(values, twiddle_real, twiddle_imag, codec_config, add_config,
                      timesteps=TIMESTEP, twiddle_dim=TWIDDLE_DIM):
    """Stream the same butterfly from bare encode, mul_gaines, and add_scale instances."""
    device = values[0].device
    lane = twiddle_real.numel()
    encoders = [encode(dict(codec_config, dim=dim)).to(device) for dim in (1, 2, 3, 4)]
    twiddle_encode = encode(dict(codec_config, dim=twiddle_dim)).to(device)
    multiply = mul_gaines({'polarity': codec_config['polarity']}).to(device)
    adder = add_scale(dict(add_config)).to(device)
    twiddle = torch.cat([twiddle_real, twiddle_imag]).view(2 * lane, 1)
    outputs = None
    for _ in range(timesteps):
        x0r, x0i, x1r, x1i = (encoder(value) for encoder, value in zip(encoders, values))
        twiddle_bit = twiddle_encode(twiddle)
        w_r = twiddle_bit.narrow(0, 0, lane)
        w_i = twiddle_bit.narrow(0, lane, lane)
        p_rr = multiply(x1r, w_r)
        p_ri = multiply(x1i, w_r)
        p_ir = multiply(x1r, w_i)
        p_ii = multiply(x1i, w_i)
        # The bias constants absorb the complemented spikes of the two negated products.
        y0r_sum = x0r + p_rr - p_ii + 1
        y0i_sum = x0i + p_ri + p_ir
        y1r_sum = x0r - p_rr + p_ii + 1
        y1i_sum = x0i - p_ri - p_ir + 2
        y_spike = adder(torch.cat([y0r_sum, y0i_sum, y1r_sum, y1i_sum], 0), entry=3, dim=None)
        outputs = tuple(y_spike.narrow(0, index * lane, lane) for index in range(4))
    return outputs


def _assert_spikes(outputs):
    for output in outputs:
        assert torch.equal(output, output.round())
        assert output.min().item() >= 0 and output.max().item() <= 1


def test_butterfly_mix_streaming():
    """Verify the bipolar-only Gaines butterfly matches its bare cores bit-exactly per device."""
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
        operation = butterfly_mix(
            twiddle_real, twiddle_imag, codec_config, add_config
        ).to(device)
        assert operation.streaming is True
        assert operation.lane == LANE
        assert operation.compensation == SCALE
        # The class encodes the constant twiddle itself, because an XNOR gate needs
        # both operands as streams.
        assert operation.internal_encode is True
        assert isinstance(operation.mul_wx, mul_gaines)
        assert operation.reference_encode.num_seq.shape == (TIMESTEP,)
        # The spike ports are the whole point of this class, so its interface
        # must declare rate coding rather than the hub family's empty mapping.
        ports = ('x0r', 'x0i', 'x1r', 'x1i', 'y0r', 'y0i', 'y1r', 'y1i')
        assert operation.encoding_io == {port: 'rc' for port in ports}
        assert operation.polarity_io == {port: 'bipolar' for port in ports}
        # The data ports carry no correlation requirement, because the only
        # correlation-sensitive operand pair is internal to the class.
        assert operation.correlation_i == {}
        assert not hasattr(operation, 'encoder_x')
        assert not hasattr(operation, 'decoder_y')

        outputs, _ = _stream(operation, values, codec_config)
        first = tuple(value.detach().clone() for value in outputs)
        _assert_spikes(first)
        assert all(value.shape == values[0].shape for value in first)
        assert operation.add_y.accumulator.shape == (4 * LANE,) + values[0].shape[1:]

        # Gaines correlation error is systematic, so the composed class is checked
        # bit-exactly against the same computation wired from bare cores.
        bare = _bare_core_stream(values, twiddle_real, twiddle_imag, codec_config, add_config)
        for name, candidate, expected in zip(('y0r', 'y0i', 'y1r', 'y1i'), first, bare):
            assert torch.equal(candidate, expected), name
            print(f'[{device}][{name}] bit-exact against the bare cores')
        assert operation.timestep_cur == TIMESTEP
        assert operation.reference_encode.timestep_cur == TIMESTEP
        assert operation.mul_wx.timestep_cur == TIMESTEP
        assert operation.add_y.timestep_cur == TIMESTEP

        operation.reset()
        assert operation.timestep_cur == 0
        assert operation.reference_encode.timestep_cur == 0
        assert operation.mul_wx.timestep_cur == 0
        assert operation.add_y.timestep_cur == 0
        assert torch.count_nonzero(operation.add_y.accumulator) == 0
        replay, _ = _stream(operation, values, codec_config)
        assert all(torch.equal(before, after) for before, after in zip(first, replay))

        # The optional dim key must reach the twiddle encoder, so a non-default
        # dimension is checked against a bare core encoding on that same dimension.
        custom_dim = TWIDDLE_DIM + 2
        custom = butterfly_mix(
            twiddle_real, twiddle_imag, dict(codec_config, dim=custom_dim), add_config
        ).to(device)
        custom_outputs, _ = _stream(custom, values, codec_config)
        custom_bare = _bare_core_stream(values, twiddle_real, twiddle_imag, codec_config,
                                        add_config, twiddle_dim=custom_dim)
        for name, candidate, expected in zip(('y0r', 'y0i', 'y1r', 'y1i'),
                                             custom_outputs, custom_bare):
            assert torch.equal(candidate, expected), name
        assert any(not torch.equal(left, right)
                   for left, right in zip(custom_outputs, first))
        print(f'[{device}] bit-exact against the bare cores at dim={custom_dim}')

        performance_operation = butterfly_mix(
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


def test_butterfly_mix_known_answer():
    """Verify the twiddle 1 - 1j, whose streams are constant, gives the exact butterfly."""
    codec_config, add_config = _configs()
    reference = butterfly_fp()
    for device in devices():
        # The twiddle 1 - 1j encodes to an all-ones and an all-zeros stream, so the XNOR
        # products are exact copies and complements of the x1 streams and no correlation
        # error arises. These four inputs and their four outputs all sit on the decoded
        # grid, 2 / TIMESTEP for the inputs and 2 * SCALE / TIMESTEP for the outputs, and
        # the accumulator drains to zero over the full period, so the result is exact and
        # the tolerance is zero.
        values = tuple(
            torch.full((4, 1), fill, dtype=global_config.ntype, device=device)
            for fill in (0.1875, 0.1875, 0.375, 0.1875)
        )
        twiddle_real = torch.ones(4, dtype=global_config.ntype, device=device)
        twiddle_imag = -torch.ones(4, dtype=global_config.ntype, device=device)
        operation = butterfly_mix(
            twiddle_real, twiddle_imag, codec_config, add_config
        ).to(device)
        _, decoded = _stream(operation, values, codec_config)
        expected = reference.to(device)(
            *values, twiddle_real.view(4, 1), twiddle_imag.view(4, 1)
        )
        for name, value, target in zip(('y0r', 'y0i', 'y1r', 'y1i'), decoded, expected):
            assert torch.equal(value, target), name
            print(f'[{device}] {name} exact at {target.flatten()[0].item():.6f}')


def test_butterfly_mix_rejects_invalid_config():
    """Verify construction and calls reject unipolar, bad twiddles, and mismatched shapes."""
    codec_config, add_config = _configs()
    twiddle = torch.zeros(4, dtype=global_config.ntype)

    unipolar = dict(codec_config, polarity='unipolar')
    try:
        butterfly_mix(twiddle, twiddle, unipolar, dict(add_config, polarity='unipolar'))
    except AssertionError as error:
        assert str(error) == "Invalid polarity: <unipolar>; legal values: <['bipolar']>.", error
    else:
        raise AssertionError('butterfly_mix accepted a unipolar configuration')

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
            butterfly_mix(bad_real, bad_imag, codec_config, add_config)
        except AssertionError as error:
            assert str(error) == expected, error
        else:
            raise AssertionError(f'butterfly_mix accepted {label} shape {tuple(value.shape)}')

    try:
        butterfly_mix('twiddle', twiddle, codec_config, add_config)
    except AssertionError as error:
        assert str(error) == (
            'Invalid twiddle_real: <str>; legal values: a non-empty 1-D tensor.'
        ), error
    else:
        raise AssertionError('butterfly_mix accepted a non-tensor twiddle_real')

    try:
        butterfly_mix(twiddle, torch.zeros(2, dtype=global_config.ntype),
                      codec_config, add_config)
    except AssertionError as error:
        assert str(error) == (
            'butterfly_mix twiddle_real length <4> must equal twiddle_imag length <2>.'
        ), error
    else:
        raise AssertionError('butterfly_mix accepted mismatched twiddle lengths')

    try:
        butterfly_mix(twiddle, twiddle, codec_config, dict(add_config, scale=10, intwidth=4))
    except AssertionError as error:
        assert str(error) == (
            'add_scale scale <10.0> exceeds accumulator maximum <7.0> for intwidth <4> '
            'and fracwidth <0>.'
        ), error
    else:
        raise AssertionError('butterfly_mix accepted scale 10 above the width-4 maximum')

    try:
        butterfly_mix(twiddle, twiddle, codec_config, dict(add_config, scale=6, intwidth=4))
    except AssertionError as error:
        assert str(error) == (
            'butterfly_mix accumulator maximum <7> raw units too small for fan-in <3> and '
            'scale <6.0>: for this bipolar-only adder acc_max must be >= (scale_raw - '
            'grid) + delta_max, with grid <0.5> and delta_max <4.5> in raw units of '
            '<1.0>, and acc_max + 1 must be > entry when scale < entry, or partial sums '
            'saturate. Increase intwidth.'
        ), error
    else:
        raise AssertionError('butterfly_mix accepted an accumulator width below the bound')

    operation = butterfly_mix(twiddle, twiddle, codec_config, add_config)
    spike = torch.zeros(4, 1, dtype=global_config.stype)
    operation(spike, spike, spike, spike)
    timestep = operation.timestep_cur
    accumulator = operation.add_y.accumulator.clone()
    for bad, message in (
        ((spike, torch.zeros(4, 2, dtype=global_config.stype), spike, spike),
         'butterfly_mix input shapes must match: got <torch.Size([4, 1])>, '
         '<torch.Size([4, 2])>, <torch.Size([4, 1])>, and <torch.Size([4, 1])>.'),
        ((torch.zeros(3, 1, dtype=global_config.stype),) * 4,
         'butterfly_mix first input dimension must equal lane <4>: '
         'got shape <torch.Size([3, 1])>.'),
    ):
        try:
            operation(*bad)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'butterfly_mix accepted input shapes {[b.shape for b in bad]}')
    # The rejection happens after __call__ ticks, so only the adder must stand still.
    assert operation.timestep_cur == timestep + 2
    assert torch.equal(operation.add_y.accumulator, accumulator)


if __name__ == '__main__':
    test_butterfly_mix_rejects_invalid_config()
    test_butterfly_mix_known_answer()
    test_butterfly_mix_streaming()
    print('Test passed.')

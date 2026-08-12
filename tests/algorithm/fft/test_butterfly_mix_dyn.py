"""Test the dynamic-scale Gaines spike-port butterfly; gradients are exempt because it has no parameters."""

import math

import torch

from napl.sim.algorithm.fft import butterfly_fp, butterfly_mix, butterfly_mix_dyn
from napl.sim.base import global_config, napl_base
from napl.sim.operation import add_scale, add_scale_dyn, decode, encode, mul_gaines
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices


LANE = 512
TIMESTEP = 256
WIDTH = int(math.log2(TIMESTEP))
SCALE_MAX = 4
SCALE = 3
TWIDDLE_DIM = 5


def _configs(scale_max=SCALE_MAX, width=WIDTH + 1):
    codec_config = {
        'polarity': 'bipolar',
        'timestep': TIMESTEP,
        'generator': 'sobol',
    }
    add_config = {
        'polarity': 'bipolar',
        'scale_max': scale_max,
        'intwidth': width,
        'fracwidth': 0,
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


def _bare_core_stream(values, twiddle_real, twiddle_imag, codec_config, add_config, scale,
                      timesteps=TIMESTEP, twiddle_dim=TWIDDLE_DIM):
    """Stream the same butterfly from bare encode, mul_gaines, and add_scale_dyn instances."""
    device = values[0].device
    lane = twiddle_real.numel()
    encoders = [encode(dict(codec_config, dim=dim)).to(device) for dim in (1, 2, 3, 4)]
    twiddle_encode = encode(dict(codec_config, dim=twiddle_dim)).to(device)
    multiply = mul_gaines({'polarity': codec_config['polarity']}).to(device)
    adder = add_scale_dyn(dict(add_config)).to(device)
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
        y_spike = adder(torch.cat([y0r_sum, y0i_sum, y1r_sum, y1i_sum], 0), scale,
                        entry=3, dim=None)
        outputs = tuple(y_spike.narrow(0, index * lane, lane) for index in range(4))
    return outputs


def test_butterfly_mix_dyn_streaming():
    """Verify the dynamic Gaines butterfly matches its bare cores bit-exactly per device."""
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
        operation = butterfly_mix_dyn(
            twiddle_real, twiddle_imag, codec_config, add_config
        ).to(device)
        assert isinstance(operation, napl_base) and not isinstance(operation, butterfly_mix)
        assert isinstance(operation.add_y, add_scale_dyn)
        assert not isinstance(operation.add_y, add_scale)
        assert isinstance(operation.mul_wx, mul_gaines)
        assert operation.scale_max == SCALE_MAX
        assert operation.compensation is None
        # The class encodes the constant twiddle itself, because an XNOR gate needs
        # both operands as streams.
        assert operation.internal_encode is True
        assert operation.reference_encode.num_seq.shape == (TIMESTEP,)
        assert operation.streaming is True
        assert operation.lane == LANE
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

        outputs, _ = _stream(operation, values, codec_config, SCALE)
        first = tuple(value.detach().clone() for value in outputs)
        for output in first:
            assert torch.equal(output, output.round())
            assert output.min().item() >= 0 and output.max().item() <= 1
        assert operation.compensation == SCALE
        assert all(value.shape == values[0].shape for value in first)
        assert operation.add_y.accumulator.shape == (4 * LANE,) + values[0].shape[1:]

        # Gaines correlation error is systematic, so the composed class is checked
        # bit-exactly against the same computation wired from bare cores.
        bare = _bare_core_stream(
            values, twiddle_real, twiddle_imag, codec_config, add_config, SCALE
        )
        for name, candidate, expected in zip(('y0r', 'y0i', 'y1r', 'y1i'), first, bare):
            assert torch.equal(candidate, expected), name
            print(f'[{device}][{name}] bit-exact against the bare cores')
        assert operation.timestep_cur == TIMESTEP
        assert operation.reference_encode.timestep_cur == TIMESTEP
        assert operation.add_y.timestep_cur == TIMESTEP

        operation.reset()
        assert operation.timestep_cur == 0
        assert operation.compensation is None
        assert operation.reference_encode.timestep_cur == 0
        assert operation.add_y.timestep_cur == 0
        assert torch.count_nonzero(operation.add_y.accumulator) == 0
        replay, _ = _stream(operation, values, codec_config, SCALE)
        assert all(torch.equal(before, after) for before, after in zip(first, replay))

        # The optional dim key must reach the twiddle encoder, so a non-default
        # dimension is checked against a bare core encoding on that same dimension.
        custom_dim = TWIDDLE_DIM + 2
        custom = butterfly_mix_dyn(
            twiddle_real, twiddle_imag, dict(codec_config, dim=custom_dim), add_config
        ).to(device)
        custom_outputs, _ = _stream(custom, values, codec_config, SCALE)
        custom_bare = _bare_core_stream(values, twiddle_real, twiddle_imag, codec_config,
                                        add_config, SCALE, twiddle_dim=custom_dim)
        for name, candidate, expected in zip(('y0r', 'y0i', 'y1r', 'y1i'),
                                             custom_outputs, custom_bare):
            assert torch.equal(candidate, expected), name
        assert any(not torch.equal(left, right)
                   for left, right in zip(custom_outputs, first))
        print(f'[{device}] bit-exact against the bare cores at dim={custom_dim}')

        performance_operation = butterfly_mix_dyn(
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


def test_butterfly_mix_dyn_known_answer():
    """Verify the twiddle 1 - 1j gives the exact butterfly at every runtime scale."""
    codec_config, add_config = _configs()
    reference = butterfly_fp()
    for device in devices():
        # The twiddle 1 - 1j encodes to an all-ones and an all-zeros stream, so the XNOR
        # products are exact copies and complements of the x1 streams and no correlation
        # error arises. These four inputs and their four outputs sit on the decoded grid,
        # 2 / TIMESTEP for the inputs and 2 * scale / TIMESTEP for the outputs at each
        # tested scale, and the accumulator drains to zero over the full period, so the
        # result is exact and the tolerance is zero.
        values = tuple(
            torch.full((4, 1), fill, dtype=global_config.ntype, device=device)
            for fill in (0.1875, 0.1875, 0.375, 0.1875)
        )
        twiddle_real = torch.ones(4, dtype=global_config.ntype, device=device)
        twiddle_imag = -torch.ones(4, dtype=global_config.ntype, device=device)
        expected = reference.to(device)(
            *values, twiddle_real.view(4, 1), twiddle_imag.view(4, 1)
        )
        for scale in (2, 3, 4):
            operation = butterfly_mix_dyn(
                twiddle_real, twiddle_imag, codec_config, add_config
            ).to(device)
            _, decoded = _stream(operation, values, codec_config, scale)
            assert operation.compensation == scale
            for name, value, target in zip(('y0r', 'y0i', 'y1r', 'y1i'), decoded, expected):
                assert torch.equal(value, target), (scale, name)
                print(f'[{device}][scale={scale}] {name} exact at '
                      f'{target.flatten()[0].item():.6f}')


def test_butterfly_mix_dyn_scale_change():
    """Verify a runtime scale change alters the output while state advances exactly once."""
    codec_config, add_config = _configs()
    for device in devices():
        twiddle = torch.full((4,), 0.5, dtype=global_config.ntype, device=device)
        spike = torch.ones(4, 1, dtype=global_config.stype, device=device)
        changed = butterfly_mix_dyn(
            twiddle, twiddle, codec_config, add_config
        ).to(device)
        held = butterfly_mix_dyn(
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


def test_butterfly_mix_dyn_rejects_invalid_config():
    """Verify construction and runtime scales are rejected with their exact messages."""
    codec_config, add_config = _configs()
    twiddle = torch.zeros(4, dtype=global_config.ntype)

    missing = dict(add_config)
    missing.pop('scale_max')
    try:
        butterfly_mix_dyn(twiddle, twiddle, codec_config, missing)
    except AssertionError as error:
        assert str(error) == 'Missing key <scale_max> in the dynamic adder configuration.', error
    else:
        raise AssertionError('butterfly_mix_dyn accepted an add config without scale_max')

    for scale_max in (0, -1, True, 2.0):
        try:
            butterfly_mix_dyn(
                twiddle, twiddle, codec_config, dict(add_config, scale_max=scale_max)
            )
        except AssertionError as error:
            assert str(error) == (
                f'butterfly_mix_dyn scale_max must be a positive int: got <{scale_max}>.'
            ), error
        else:
            raise AssertionError(f'butterfly_mix_dyn accepted scale_max {scale_max!r}')

    try:
        butterfly_mix_dyn(
            twiddle, twiddle, dict(codec_config, polarity='unipolar'),
            dict(add_config, polarity='unipolar')
        )
    except AssertionError as error:
        assert str(error) == "Invalid polarity: <unipolar>; legal values: <['bipolar']>.", error
    else:
        raise AssertionError('butterfly_mix_dyn accepted a unipolar configuration')

    operation = butterfly_mix_dyn(twiddle, twiddle, codec_config, add_config)
    spike = torch.zeros(4, 1, dtype=global_config.stype)
    operation(spike, spike, spike, spike, 2)
    timestep = operation.timestep_cur
    accumulator = operation.add_y.accumulator.clone()
    for scale, message in (
        (2.0, 'butterfly_mix_dyn scale must be an int: got <2.0>.'),
        (True, 'butterfly_mix_dyn scale must be an int: got <True>.'),
        ('2', "butterfly_mix_dyn scale must be an int: got <2>."),
        (0, 'butterfly_mix_dyn scale <0> outside the supported range <1> to scale_max <4.0>.'),
        (5, 'butterfly_mix_dyn scale <5> outside the supported range <1> to scale_max <4.0>.'),
    ):
        try:
            operation(spike, spike, spike, spike, scale)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'butterfly_mix_dyn accepted runtime scale {scale!r}')
    # An invalid scale raises inside __call__, before the timestep or adder advance.
    assert operation.timestep_cur == timestep
    assert torch.equal(operation.add_y.accumulator, accumulator)


if __name__ == '__main__':
    test_butterfly_mix_dyn_rejects_invalid_config()
    test_butterfly_mix_dyn_known_answer()
    test_butterfly_mix_dyn_scale_change()
    test_butterfly_mix_dyn_streaming()
    print('Test passed.')

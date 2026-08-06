import math

import torch

from napl.sim.algorithm.fft import (
    butterfly_binary,
    butterfly_spike,
)
from napl.sim.base import global_config
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices


TIMESTEP = 256
WIDTH = math.log2(TIMESTEP)
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
        'width': WIDTH + 1,
    }
    return codec_config, add_config


def _run(operation, inputs, timesteps=TIMESTEP):
    """Drive one full stream, one timestep per call."""
    outputs = None
    for _ in range(timesteps):
        outputs = operation(*inputs)
    return outputs


def test_butterfly_spike():
    """Verify the streaming complex butterfly matches the decoded FFT reference across devices."""
    torch.manual_seed(0)
    codec_config, add_config = _configs()
    inputs_cpu = tuple(
        gen_rand_tensor(
            codec_config['polarity'],
            shape=(512, 1),
            width=WIDTH,
        ).type(global_config.ntype)
        for _ in range(6)
    )

    cpu_runtime = None
    for device in devices():
        inputs = tuple(value.to(device) for value in inputs_cpu)
        reference_values = butterfly_binary().to(device)(*inputs)
        operation = butterfly_spike(
            codec_config,
            codec_config,
            add_config,
            codec_config,
        ).to(device)

        first = _run(operation, inputs)
        first = tuple(value.detach().clone() for value in first)
        reference = torch.cat(reference_values, dim=0) / SCALE
        error, _ = operation.accuracy_y.analyze(reference, verbose=True)
        assert error.pow(2).mean().sqrt() < 0.2
        # One call is one timestep, so the module counts the whole run.
        assert operation.timestep_cur == TIMESTEP

        operation.reset()
        assert operation.timestep_cur == 0
        assert operation.x_stack is None
        assert operation.encoder_x.timestep_cur == 0
        assert operation.decoder_y.timestep_cur == 0
        assert operation.accuracy_y.timestep_cur == 0
        replay = _run(operation, inputs)
        assert all(
            torch.equal(before, after)
            for before, after in zip(first, replay)
        )

        device_runtime = benchmark(
            lambda values: _run(operation, values),
            inputs_cpu,
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
            f'cpu_runtime={cpu_runtime * 1e3:.1f}ms, '
            f'speedup={cpu_runtime / device_runtime:.2f}x'
        )


def test_butterfly_known_answer():
    """Verify a zero twiddle passes the first input through at the adder scale."""
    codec_config, add_config = _configs()
    for device in devices():
        zeros = torch.zeros(4, 1, dtype=global_config.ntype, device=device)
        x0r = torch.full((4, 1), 0.5, dtype=global_config.ntype, device=device)
        inputs = (x0r, zeros, zeros, zeros, zeros, zeros)
        operation = butterfly_spike(
            codec_config, codec_config, add_config, codec_config,
        ).to(device)
        y0r, y0i, y1r, y1i = _run(operation, inputs)
        # With w = 0 and x1 = 0 the twiddle term vanishes: y0 = y1 = x0 / scale.
        expected = x0r / SCALE
        for name, value in (('y0r', y0r), ('y1r', y1r)):
            torch.testing.assert_close(value, expected, atol=0.05, rtol=0)
            print(f'[{device}] {name} max_err={(value - expected).abs().max().item():.4f}')
        for name, value in (('y0i', y0i), ('y1i', y1i)):
            torch.testing.assert_close(value, zeros, atol=0.05, rtol=0)


def test_butterfly_rejects_unipolar():
    """Verify butterfly_spike rejects unipolar, which cannot represent x0 - w x1."""
    codec_config, add_config = _configs()
    unipolar_codec = dict(codec_config, polarity='unipolar')
    unipolar_add = dict(add_config, polarity='unipolar')
    try:
        butterfly_spike(unipolar_codec, unipolar_codec, unipolar_add, unipolar_codec)
    except AssertionError:
        return
    raise AssertionError('butterfly_spike accepted a unipolar configuration')


def test_butterfly_binary_is_napl_module():
    """Verify the binary reference follows the napl single-shot module contract."""
    operation = butterfly_binary()
    assert operation.streaming is False
    assert operation.layer == 'algorithm'
    operation.reset()
    assert operation.timestep_cur == 0
    inputs = tuple(torch.ones(2, 1, dtype=global_config.ntype) for _ in range(6))
    y0r, y0i, y1r, y1i = operation(*inputs)
    # t = (1*1 - 1*1, 1*1 + 1*1) = (0, 2), so y0 = (1, 3) and y1 = (1, -1).
    assert torch.equal(y0r, inputs[0])
    assert torch.equal(y0i, inputs[0] * 3)
    assert torch.equal(y1r, inputs[0])
    assert torch.equal(y1i, -inputs[0])


if __name__ == '__main__':
    test_butterfly_binary_is_napl_module()
    test_butterfly_rejects_unipolar()
    test_butterfly_known_answer()
    test_butterfly_spike()
    print('Test passed.')

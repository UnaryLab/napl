"""
Per-device (cpu/cuda/mps) test of the stability builder: the emitted stream
decodes back to the source value within threshold, its measured stability tracks
the requested normalized stability, and reset restores the initial stream
exactly. Timing is absolute (no baseline exists).
"""
import time

import torch

from napl.metric import stability
from napl.metric.stability_builder import stability_builder
from napl.utils import devices, gen_rand_tensor, sync


def run_stream(builder, timestep):
    total = None
    for _ in range(timestep):
        spike = builder().type(torch.float32)
        total = spike if total is None else total + spike
    return total


def run_decode(val, device, timestep, threshold):
    v = val.to(device)
    builder = stability_builder(
        val,
        {
            'polarity': 'bipolar',
            'threshold': threshold,
            'normstability': 0.8,
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 1,
        },
    ).to(device)
    ones = run_stream(builder, timestep)
    decoded = ones.div(timestep).mul(2).sub(1)
    return builder, (decoded - v).abs()


def run_measured(val, device, timestep, threshold, normstability):
    v = val.to(device)
    builder = stability_builder(
        val,
        {
            'polarity': 'bipolar',
            'threshold': threshold,
            'normstability': normstability,
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 1,
        },
    ).to(device)
    stab = stability(
        v, {'polarity': 'bipolar', 'threshold': threshold}
    ).to(device)
    for _ in range(timestep):
        stab(builder().type(stab.stype))
    return stab.analyze()[0]


def test_fidelity():
    timestep = 256
    threshold = 0.05
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        _, error = run_decode(val, device, timestep, threshold)
        print(
            f'[{device}] decode err mean={error.mean().item():.4f}, '
            f'max={error.max().item():.4f}'
        )
        assert error.max() <= threshold + 2.0 / timestep, error.max()


def test_known_answer():
    timestep = 256
    threshold = 0.05
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        means = []
        for normstability in [0.1, 0.9]:
            result = run_measured(
                val, device, timestep, threshold, normstability
            )
            assert result.min() >= 0.0 and result.max() <= 1.0, (
                result.min(),
                result.max(),
            )
            means.append(result.mean().item())
            print(
                f'[{device}] normstability={normstability:.1f} -> '
                f'measured stability mean={means[-1]:.4f}'
            )
        assert means[1] > means[0], means


def test_reset():
    timestep = 256
    threshold = 0.05
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        builder, _ = run_decode(val, device, timestep, threshold)
        builder.reset()
        assert not builder.valid
        first = builder()
        builder.reset()
        assert torch.equal(first, builder()), 'reset did not restore the stream'


def test_performance():
    timestep = 256
    threshold = 0.05
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        builder, _ = run_decode(val, device, timestep, threshold)
        builder.reset()
        sync(device)
        start = time.time()
        run_stream(builder, timestep)
        sync(device)
        elapsed = time.time() - start
        print(
            f'[{device}] {timestep} timesteps in {elapsed * 1e3:.1f} ms '
            f'({timestep / elapsed:.0f} steps/s)'
        )


if __name__ == '__main__':
    test_fidelity()
    test_known_answer()
    test_reset()
    test_performance()
    print('Test passed.')

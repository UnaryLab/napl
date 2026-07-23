"""
Per-device (cpu/cuda/mps) test of the stability metric: per-element stability of
a sobol-coded stream is in [0, 1] with a high mean, and a reset re-run
reproduces the identical result. Timing is absolute (no baseline exists).
"""
import time

import torch

from napl.metric import stability
from napl.module import encoder
from napl.utils import devices, gen_rand_tensor, sync


def run_stability(val, device, timestep, modules=None):
    v = val.to(device)
    if modules is None:
        cfg = {
            'polarity': 'bipolar',
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 1,
        }
        modules = (
            encoder(cfg).to(device),
            stability(v, {'polarity': 'bipolar', 'threshold': 0.05}).to(device),
        )
    enc, stab = modules

    sync(device)
    start = time.time()
    for _ in range(timestep):
        stab(enc(v))
    result = stab.analyze()[0].detach().cpu().clone()
    sync(device)
    elapsed = time.time() - start
    return result, elapsed, modules


def test_fidelity():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        result, _, _ = run_stability(val, device, timestep)
        assert result.min() >= 0.0 and result.max() <= 1.0, (
            result.min(),
            result.max(),
        )
        assert result.mean() > 0.3, result.mean()
        print(
            f'[{device}] stability mean={result.mean().item():.4f}, '
            f'min={result.min().item():.4f}, max={result.max().item():.4f}'
        )


def test_reset():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        first, _, modules = run_stability(val, device, timestep)
        enc, stab = modules
        enc.reset()
        stab.reset()
        assert not stab.valid
        second, _, _ = run_stability(val, device, timestep, modules)
        assert torch.equal(first, second), 'reset re-run diverged'


def test_performance():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        _, _, modules = run_stability(val, device, timestep)
        for module in modules:
            module.reset()
        result, elapsed, _ = run_stability(val, device, timestep, modules)
        print(
            f'[{device}] stability mean={result.mean().item():.4f}, '
            f'time={elapsed * 1000:.1f}ms'
        )


if __name__ == '__main__':
    test_fidelity()
    test_reset()
    test_performance()
    print('Test passed.')

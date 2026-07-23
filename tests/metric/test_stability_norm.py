"""
Per-device (cpu/cuda/mps) test of the normalized stability metric: per-element
values in [0, 1] with a high mean, cross-device agreement on identical inputs, a
constant stream normalizing to exactly 1, and a reset re-run reproducing the
identical result. Timing is absolute (no baseline exists).
"""
import time

import torch

from napl.metric.stability_norm import stability_norm
from napl.module import encoder
from napl.utils import devices, gen_rand_tensor, sync


def run_stability_norm(val, device, timestep, modules=None):
    val = val.to(device)
    if modules is None:
        cfg = {
            'polarity': 'bipolar',
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 1,
        }
        modules = (
            encoder(cfg).to(device),
            stability_norm(
                val, {'polarity': 'bipolar', 'threshold': 0.05}
            ).to(device),
        )
    enc, stab_norm = modules

    sync(device)
    start = time.time()
    for _ in range(timestep):
        stab_norm(enc(val))
    result = stab_norm.analyze()[0].detach().cpu().clone()
    sync(device)
    elapsed = time.time() - start
    return result, elapsed, modules


def test_fidelity():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)
    results = {}

    for device in devices():
        results[device], _, _ = run_stability_norm(val, device, timestep)
        print(
            f'[{device}] stability_norm mean={results[device].mean().item():.4f}, '
            f'min={results[device].min().item():.4f}, '
            f'max={results[device].max().item():.4f}'
        )
        assert results[device].min() >= 0.0 and results[device].max() <= 1.0
        assert results[device].mean() > 0.3, results[device].mean()
        assert torch.allclose(results[device], results['cpu'], atol=1e-5), (
            f'{device} disagrees with cpu'
        )


def test_known_answer():
    timestep = 256

    for device in devices():
        ones = torch.ones(4, device=device)
        stab_norm = stability_norm(
            ones, {'polarity': 'bipolar', 'threshold': 0.05}
        ).to(device)
        for _ in range(timestep):
            stab_norm(ones)
        assert torch.all(stab_norm.analyze()[0] == 1.0)


def test_reset():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        first, _, modules = run_stability_norm(val, device, timestep)
        enc, stab_norm = modules
        enc.reset()
        stab_norm.reset()
        assert not stab_norm.valid
        assert (
            stab_norm.timestep_cur == 0
            and stab_norm.stability_norm.abs().sum() == 0
        )
        second, _, _ = run_stability_norm(val, device, timestep, modules)
        assert torch.equal(first, second), 'reset re-run diverged'


def test_performance():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)
    times = {}

    for device in devices():
        _, _, modules = run_stability_norm(val, device, timestep)
        for module in modules:
            module.reset()
        result, times[device], _ = run_stability_norm(
            val, device, timestep, modules
        )
        print(
            f'[{device}] stability_norm mean={result.mean().item():.4f}, '
            f'time={times[device]:.3f}s '
            f'(vs cpu {times["cpu"] / times[device]:.2f}x)'
        )


if __name__ == '__main__':
    test_fidelity()
    test_known_answer()
    test_reset()
    test_performance()
    print('Test passed.')

"""
Per-device (cpu/cuda/mps) test of the correlation (SCC) metric: a stream vs an
identical copy scores +1, vs its complement -1, vs an independent stream near 0,
and a reset re-run reproduces the identical result. Timing is absolute (no
baseline exists).
"""
import time

import torch

from napl.metric import correlation
from napl.module import encoder
from napl.utils import devices, gen_rand_tensor, sync


def make_modules(device, timestep):
    cfg = {
        'polarity': 'bipolar',
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 1,
    }
    cfg_indep = dict(cfg, dim=2)
    return (
        encoder(cfg).to(device),
        encoder(cfg).to(device),
        encoder(cfg_indep).to(device),
        correlation().to(device),
        correlation().to(device),
        correlation().to(device),
    )


def run_correlation(val, device, timestep, modules=None):
    v = val.to(device)
    if modules is None:
        modules = make_modules(device, timestep)
    enc_a, enc_b, enc_indep, corr_self, corr_inv, corr_indep = modules

    sync(device)
    start = time.time()
    for _ in range(timestep):
        spike_a = enc_a(v)
        spike_b = enc_b(v)
        spike_indep = enc_indep(v)
        corr_self(spike_a, spike_b)
        corr_inv(spike_a, 1 - spike_a)
        corr_indep(spike_a, spike_indep)
    result = (
        corr_self.analyze()[0].detach().cpu().clone(),
        corr_inv.analyze()[0].detach().cpu().clone(),
        corr_indep.analyze()[0].detach().cpu().clone(),
    )
    sync(device)
    elapsed = time.time() - start
    return result, elapsed, modules


def test_fidelity():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        (_, _, scc_indep), _, _ = run_correlation(val, device, timestep)
        assert scc_indep.abs().mean() < 0.2, scc_indep.abs().mean()
        print(f'[{device}] SCC independent={scc_indep.mean().item():.4f}')


def test_known_answer():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        (scc_self, scc_inv, _), _, _ = run_correlation(val, device, timestep)
        assert scc_self.mean() > 0.99, scc_self.mean()
        assert scc_inv.mean() < -0.99, scc_inv.mean()
        print(
            f'[{device}] SCC self={scc_self.mean().item():.4f}, '
            f'inv={scc_inv.mean().item():.4f}'
        )


def test_reset():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        first, _, modules = run_correlation(val, device, timestep)
        for module in modules:
            module.reset()
        corr_self, corr_inv, corr_indep = modules[-3:]
        assert not (corr_self.valid or corr_inv.valid or corr_indep.valid)
        second, _, _ = run_correlation(val, device, timestep, modules)
        for before, after in zip(first, second):
            assert torch.equal(before, after), 'reset re-run diverged'


def test_performance():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        _, _, modules = run_correlation(val, device, timestep)
        for module in modules:
            module.reset()
        (scc_self, scc_inv, scc_indep), elapsed, _ = run_correlation(
            val, device, timestep, modules
        )
        print(
            f'[{device}] SCC self={scc_self.mean().item():.4f}, '
            f'inv={scc_inv.mean().item():.4f}, '
            f'indep={scc_indep.mean().item():.4f}, '
            f'time={elapsed * 1000:.1f}ms'
        )


if __name__ == '__main__':
    test_fidelity()
    test_known_answer()
    test_reset()
    test_performance()
    print('Test passed.')

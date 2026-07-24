"""
Per-device (cpu/cuda/mps) test of the normalized stability metric: per-element
values in [0, 1] with a high mean, cross-device agreement on identical inputs, a
constant stream normalizing to exactly 1, and a reset re-run reproducing the
identical result. Performance is compared with the same metric on CPU.
"""
import torch

from napl.sim.metric.stability_norm import stability_norm
from napl.sim.module import encoder
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices, timer


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
                val.cpu(), {'polarity': 'bipolar', 'threshold': 0.05}
            ).to(device),
        )
    enc, stab_norm = modules
    assert stab_norm.source.device.type == device
    assert stab_norm.stability.source.device.type == device

    with timer(device) as elapsed:
        for _ in range(timestep):
            stab_norm(enc(val))
    result = stab_norm.analyze()[0].detach().cpu().clone()
    return result, elapsed.seconds, modules


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
        for polarity in ('unipolar', 'bipolar'):
            source = torch.ones(4)
            ones = source.to(device)
            stab_norm = stability_norm(
                source, {'polarity': polarity, 'threshold': 0.05}
            ).to(device)
            assert stab_norm.source.device.type == device
            assert not stab_norm.valid
            for _ in range(timestep):
                stab_norm(ones)
            result, max_index = stab_norm.analyze()
            assert stab_norm.valid
            assert stab_norm.timestep_cur == timestep
            assert result.shape == ones.shape
            assert result.dtype == ones.dtype
            assert torch.equal(result, torch.ones_like(result))
            assert max_index.item() == 0

        boundary = stability_norm(
            torch.ones(1),
            {'polarity': 'bipolar', 'threshold': 0.05},
        ).to(device)
        boundary(torch.ones(1, device=device))
        assert torch.equal(
            boundary.analyze()[0],
            torch.zeros(1, device=device),
        )


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
        assert stab_norm.stability.timestep_cur == 0
        assert stab_norm.stability.accuracy.timestep_cur == 0
        second, _, _ = run_stability_norm(val, device, timestep, modules)
        assert torch.equal(first, second), 'reset re-run diverged'


def test_performance():
    timestep = 64
    torch.manual_seed(0)
    stream = torch.randint(0, 2, (timestep, 256)).float()
    val = gen_rand_tensor('bipolar', shape=(256,), width=6)

    cpu_runtime = None
    for device in devices():
        metric = stability_norm(
            val,
            {'polarity': 'bipolar', 'threshold': 0.05},
        ).to(device)
        inputs = (stream,)

        def run(target_metric, values):
            spikes = values[0]
            for spike in spikes:
                target_metric(spike)

        device_runtime = benchmark(
            lambda values: run(metric, values),
            inputs,
            device,
            warmup_runs=1,
            trials=3,
            prepare=metric.reset,
        )
        metric.analyze()
        if device == 'cpu':
            cpu_runtime = device_runtime
        assert cpu_runtime is not None
        print(
            f'[{device}] device_runtime={device_runtime * 1e3:.1f}ms, '
            f'cpu_runtime={cpu_runtime * 1e3:.1f}ms, '
            f'speedup={cpu_runtime / device_runtime:.2f}x'
        )


if __name__ == '__main__':
    test_known_answer()
    test_fidelity()
    test_reset()
    test_performance()
    print('Test passed.')

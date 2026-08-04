"""
Per-device (cpu/cuda/mps) test of the stability metric: per-element stability of
a sobol-coded stream is in [0, 1] with a high mean, and a reset re-run
reproduces the identical result. Performance is compared with the same metric
on CPU.
"""
import torch

from napl.sim.metric import stability
from napl.sim.operation import encode
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices, timer


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
            encode(cfg).to(device),
            stability(
                val, {'polarity': 'bipolar', 'threshold': 0.05}
            ).to(device),
        )
    enc, stab = modules
    assert stab.source.device.type == device

    with timer(device) as elapsed:
        for _ in range(timestep):
            stab(enc(v))
    result = stab.analyze()[0].detach().cpu().clone()
    return result, elapsed.seconds, modules


def test_fidelity():
    """Verify stability scores remain bounded and identify convergent bipolar streams."""
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(20, 50), width=8)

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


def test_known_answer():
    """Verify stability reports exact cycle-to-stable values for a known stream."""
    timestep = 4
    source = torch.ones(2)
    spike = torch.tensor([1.0, 0.0])

    for device in devices():
        for polarity in ('unipolar', 'bipolar'):
            metric = stability(
                source,
                {'polarity': polarity, 'threshold': 0.05},
            ).to(device)
            assert metric.source.device.type == device
            assert not metric.valid
            for _ in range(timestep):
                metric(spike.to(device))
            result, analysis_result = metric.analyze()
            expected = torch.tensor([0.75, 0.0], device=device)
            assert metric.valid
            assert metric.timestep_cur == timestep
            assert result.shape == source.shape
            assert result.dtype == source.dtype
            assert torch.equal(result, expected)
            assert analysis_result.max_absolute_index.item() == 0

        metric = stability(
            torch.ones(1),
            {'polarity': 'bipolar', 'threshold': 0.05},
        ).to(device)
        metric(torch.ones(1, device=device))
        assert torch.equal(
            metric.analyze()[0],
            torch.zeros(1, device=device),
        )


def test_reset():
    """Verify resetting stability clears nested state and reproduces its result."""
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        first, _, modules = run_stability(val, device, timestep)
        enc, stab = modules
        enc.reset()
        stab.reset()
        assert not stab.valid
        assert stab.timestep_cur == 0
        assert stab.accuracy.timestep_cur == 0
        assert stab.cycle_to_stable.abs().sum() == 0
        second, _, _ = run_stability(val, device, timestep, modules)
        assert torch.equal(first, second), 'reset re-run diverged'


def test_performance():
    """Verify stability updates meet the configured runtime bounds across supported devices."""
    timestep = 256
    torch.manual_seed(0)
    stream = torch.randint(0, 2, (timestep, 1000000)).float()
    source = gen_rand_tensor('bipolar', shape=(1000000,), width=8)

    cpu_runtime = None
    for device in devices():
        metric = stability(
            source,
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

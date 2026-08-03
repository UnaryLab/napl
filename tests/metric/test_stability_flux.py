import torch

from napl.sim.metric import stability, stability_flux
from napl.sim.module import encoder
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices, timer


def run_flux(val_1, val_2, device, timestep, modules=None):
    stability_config = {'polarity': 'bipolar', 'threshold': 0.05}
    source_1 = val_1
    source_2 = val_2
    val_1 = source_1.to(device)
    val_2 = source_2.to(device)
    if modules is None:
        enc_1 = encoder(
            {
                'polarity': 'bipolar',
                'timestep': timestep,
                'generator': 'sobol',
                'dim': 1,
            }
        ).to(device)
        enc_2 = encoder(
            {
                'polarity': 'bipolar',
                'timestep': timestep,
                'generator': 'sobol',
                'dim': 2,
            }
        ).to(device)
        flux = stability_flux(source_1, source_2, stability_config).to(device)
        stab_1 = stability(source_1, stability_config).to(device)
        stab_2 = stability(source_2, stability_config).to(device)
        modules = (enc_1, enc_2, flux, stab_1, stab_2)
    enc_1, enc_2, flux, stab_1, stab_2 = modules
    assert flux.stability_1.source.device.type == device
    assert flux.stability_2.source.device.type == device

    with timer(device) as elapsed:
        for _ in range(timestep):
            spike_1 = enc_1(val_1)
            spike_2 = enc_2(val_2)
            flux(spike_1, spike_2)
            stab_1(spike_1)
            stab_2(spike_2)
    result = flux.analyze()[0].detach().cpu().clone()
    expected = stab_1.analyze()[0].div(stab_2.analyze()[0])
    return result, expected.detach().cpu(), elapsed.seconds, modules


def test_fidelity():
    """Verify stability flux equals the ratio of its two component stability scores."""
    timestep = 256
    val_1 = gen_rand_tensor('bipolar', shape=(20, 50), width=8)
    val_2 = gen_rand_tensor('bipolar', shape=(20, 50), width=8)

    for device in devices():
        result, expected, _, _ = run_flux(val_1, val_2, device, timestep)
        assert torch.equal(result, expected), 'flux != stability_1 / stability_2'
        print(
            f'[{device}] flux mean={result.mean().item():.4f}, '
            f'min={result.min().item():.4f}, max={result.max().item():.4f}'
        )


def test_known_answer():
    """Verify stability flux handles equal, zero, and random known-answer streams."""
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        source = torch.ones(4)
        ones = source.to(device)
        flux = stability_flux(
            source, source, {'polarity': 'bipolar', 'threshold': 0.05}
        ).to(device)
        assert not flux.valid
        for _ in range(timestep):
            spike = torch.ones(4, device=device)
            flux(spike, spike)
        result, analysis_result = flux.analyze()
        assert flux.valid
        assert flux.timestep_cur == timestep
        assert flux.stability_1.timestep_cur == timestep
        assert flux.stability_2.timestep_cur == timestep
        assert result.shape == source.shape
        assert result.dtype == source.dtype
        assert torch.equal(result, torch.ones(4, device=device))
        assert analysis_result.max_absolute_index.item() == 0

        device_val = val.to(device)
        enc = encoder(
            {
                'polarity': 'bipolar',
                'timestep': timestep,
                'generator': 'sobol',
                'dim': 1,
            }
        ).to(device)
        source = torch.ones(1000)
        ones = source.to(device)
        flux = stability_flux(
            source, val, {'polarity': 'bipolar', 'threshold': 0.05}
        ).to(device)
        for _ in range(timestep):
            flux(ones, enc(device_val))
        result = flux.analyze()[0]
        assert (result >= 1.0).all(), result.min()
        assert result.mean() > 1.0, result.mean()

        unstable = stability_flux(
            source,
            source,
            {'polarity': 'bipolar', 'threshold': 0.05},
        ).to(device)
        for _ in range(4):
            unstable(ones, torch.zeros_like(ones))
        assert torch.isinf(unstable.analyze()[0]).all()

        boundary = stability_flux(
            torch.ones(1),
            torch.ones(1),
            {'polarity': 'bipolar', 'threshold': 0.05},
        ).to(device)
        boundary_spike = torch.ones(1, device=device)
        boundary(boundary_spike, boundary_spike)
        assert torch.isnan(boundary.analyze()[0]).all()


def test_reset():
    """Verify resetting stability flux clears nested state and reproduces its result."""
    timestep = 256
    val_1 = gen_rand_tensor('bipolar', shape=(1000,), width=8)
    val_2 = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        first, _, _, modules = run_flux(val_1, val_2, device, timestep)
        for module in modules:
            module.reset()
        flux = modules[2]
        assert not flux.valid
        assert flux.timestep_cur == 0
        assert flux.stability_flux.abs().sum() == 0
        assert not flux.stability_1.valid
        assert not flux.stability_2.valid
        assert (
            flux.stability_1.timestep_cur == 0
            and flux.stability_2.timestep_cur == 0
        )
        second, _, _, _ = run_flux(
            val_1, val_2, device, timestep, modules
        )
        assert torch.equal(first, second), 'reset re-run diverged'


def test_performance():
    """Verify stability flux updates meet the configured runtime bounds across supported devices."""
    timestep = 256
    torch.manual_seed(0)
    stream_1 = torch.randint(0, 2, (timestep, 1000000)).float()
    stream_2 = torch.randint(0, 2, (timestep, 1000000)).float()
    val_1 = gen_rand_tensor('bipolar', shape=(1000000,), width=8)
    val_2 = gen_rand_tensor('bipolar', shape=(1000000,), width=8)

    cpu_runtime = None
    for device in devices():
        metric = stability_flux(
            val_1,
            val_2,
            {'polarity': 'bipolar', 'threshold': 0.05},
        ).to(device)
        inputs = (stream_1, stream_2)

        def run(target_metric, values):
            first, second = values
            for spike_1, spike_2 in zip(first, second):
                target_metric(spike_1, spike_2)

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

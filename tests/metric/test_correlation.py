"""
Per-device (cpu/cuda/mps) test of the correlation (SCC) metric: a stream vs an
identical copy scores +1, vs its complement -1, vs an independent stream near 0,
and a reset re-run reproduces the identical result. Performance is compared
with the same metric on CPU.
"""
import torch

from napl.sim.metric import correlation
from napl.sim.module import encoder
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices, timer


def correlation_reference(stream_1, stream_2):
    in_1 = stream_1.ne(0)
    in_2 = stream_2.ne(0)
    a = (in_1 & in_2).sum(dim=0).float()
    b = in_1.sum(dim=0).float() - a
    c = in_2.sum(dim=0).float() - a
    d = stream_1.shape[0] - a - b - c
    numerator = a * d - b * c
    positive = numerator > 0
    positive_denominator = torch.minimum(a + b, a + c) * stream_1.shape[0]
    positive_denominator -= (a + b) * (a + c)
    negative_denominator = (a + b) * (a + c)
    negative_denominator -= (
        torch.maximum(a - d, torch.zeros_like(a)) * stream_1.shape[0]
    )
    return torch.where(
        positive,
        numerator / positive_denominator.clamp_min(1),
        numerator / negative_denominator.clamp_min(1),
    )


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

    with timer(device) as elapsed:
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
    return result, elapsed.seconds, modules


def test_fidelity():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(20, 50), width=8)

    for device in devices():
        (_, _, scc_indep), _, _ = run_correlation(val, device, timestep)
        assert scc_indep.abs().mean() < 0.2, scc_indep.abs().mean()
        print(f'[{device}] SCC independent={scc_indep.mean().item():.4f}')

        torch.manual_seed(0)
        stream_1 = torch.randint(
            0, 2, (timestep, 20, 50), device=device
        ).float()
        stream_2 = torch.randint(
            0, 2, (timestep, 20, 50), device=device
        ).float()
        metric = correlation().to(device)
        for spike_1, spike_2 in zip(stream_1, stream_2):
            metric(spike_1, spike_2)
        result, _ = metric.analyze()
        expected = correlation_reference(stream_1, stream_2)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_known_answer():
    stream = torch.tensor([1, 1, 0, 0]).repeat(64)
    independent = torch.tensor([1, 0, 1, 0]).repeat(64)
    cases = (
        (torch.tensor([1]), torch.tensor([1]), 0.0),
        (stream, stream, 1.0),
        (stream, 1 - stream, -1.0),
        (stream, independent, 0.0),
    )

    for device in devices():
        for stream_1, stream_2, expected in cases:
            metric = correlation().to(device)
            assert not metric.valid
            for spike_1, spike_2 in zip(stream_1, stream_2):
                metric(
                    input_1=spike_1.to(device),
                    input_2=spike_2.to(device),
                )
            result, analysis_result = metric.analyze()
            assert metric.valid
            assert metric.timestep_cur == stream_1.numel()
            assert result.shape == torch.Size([1])
            assert result.item() == expected
            assert analysis_result.max_absolute_index.item() == 0


def test_reset():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        first, _, modules = run_correlation(val, device, timestep)
        for module in modules:
            module.reset()
        corr_self, corr_inv, corr_indep = modules[-3:]
        assert not (corr_self.valid or corr_inv.valid or corr_indep.valid)
        for metric in (corr_self, corr_inv, corr_indep):
            assert metric.timestep_cur == 0
            for state in (
                metric.paired_11,
                metric.sum_1,
                metric.sum_2,
                metric.input_1_d,
            ):
                assert state.shape == torch.Size([1])
                assert state.item() == 0
        second, _, _ = run_correlation(val, device, timestep, modules)
        for before, after in zip(first, second):
            assert torch.equal(before, after), 'reset re-run diverged'


def test_performance():
    timestep = 256
    torch.manual_seed(0)
    stream_1 = torch.randint(0, 2, (timestep, 1000)).float()
    stream_2 = torch.randint(0, 2, (timestep, 1000)).float()

    cpu_runtime = None
    for device in devices():
        metric = correlation().to(device)
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

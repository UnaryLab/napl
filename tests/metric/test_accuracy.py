"""
Per-device (cpu/cuda/mps) test of the accuracy metric on the canonical mul_csg
round-trip: the progressive error vs the analytic product stays within the SC
bound, the metric agrees with the decoder, and a reset re-run reproduces the
identical result. Performance is compared with the same metric on CPU.
"""
import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.sim.metric import accuracy
from napl.sim.module import decoder, encoder
from napl.sim.operation import mul_csg
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices, timer


class napl_mul_csg(napl_base):
    def __init__(self, codec_config, mul_csg_config, acc_config):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.mul_csg = mul_csg(mul_csg_config)
        self.accuracy = accuracy(acc_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike = self.encoder(input_0)
        o_spike = self.mul_csg(i_spike, input_1)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def run_accuracy(input_0, input_1, device, timestep, model=None):
    codec_config = {
        'polarity': 'bipolar',
        'timestep': timestep,
        'generator': 'sobol',
    }
    acc_config = {
        'polarity': 'bipolar',
        'name': 'mul_csg_inst',
    }
    i_0 = input_0.to(device)
    i_1 = input_1.to(device)
    if model is None:
        model = napl_mul_csg(codec_config, codec_config, acc_config).to(device)
    reference = i_0 * i_1

    with timer(device) as elapsed:
        model(i_0, i_1, timesteps=timestep)
    spike_error, _ = model.accuracy.analyze(reference)

    return (
        model,
        reference,
        spike_error.detach().cpu().clone(),
        model.accuracy.spike_value.detach().cpu().clone(),
        elapsed.seconds,
    )


def make_inputs(timestep):
    input_0 = gen_rand_tensor(
        'bipolar', shape=(100, 100), width=math.log2(timestep)
    ).type(global_config.ntype)
    input_1 = gen_rand_tensor(
        'bipolar', shape=(100, 100), width=math.log2(timestep)
    ).type(global_config.ntype)
    return input_0, input_1


def test_known_answer():
    """Verify accuracy decoding and error statistics for known unipolar and bipolar streams."""
    cases = (
        (
            'unipolar',
            (torch.tensor([1.0]),),
            torch.tensor([1.0]),
        ),
        (
            'unipolar',
            (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])),
            torch.tensor([1.0, 0.5]),
        ),
        (
            'bipolar',
            (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0])),
            torch.tensor([1.0, 0.0]),
        ),
    )

    for device in devices():
        for polarity, spikes, expected in cases:
            metric = accuracy({'polarity': polarity}).to(device)
            assert not metric.valid
            for spike in spikes:
                metric(spike.to(device))
            error, result = metric.analyze(expected.to(device))
            assert metric.valid
            assert metric.timestep_cur == len(spikes)
            assert error.shape == expected.shape
            assert error.dtype == expected.dtype
            assert torch.equal(error, torch.zeros_like(error))
            assert result.max_absolute_index.item() == 0


def test_scaled_reference():
    """Verify reference scaling changes accuracy errors without changing decoded spike values."""
    reference = torch.tensor([2.0, 1.0])

    for device in devices():
        metric = accuracy({'polarity': 'unipolar'}).to(device)
        device_reference = reference.to(device)
        metric(torch.ones(2, device=device))

        unscaled, _ = metric.analyze(device_reference)
        unscaled = unscaled.detach().clone()
        scaled, result = metric.analyze(
            device_reference,
            scale_ref=2.0,
        )

        assert torch.equal(unscaled, torch.tensor([-1.0, 0.0], device=device))
        assert torch.equal(scaled, torch.tensor([0.0, 0.5], device=device))
        assert result.max_absolute_index.item() == 1
        assert torch.equal(device_reference, reference.to(device))


def test_fidelity():
    """Verify accuracy matches decoder values and product references within stochastic tolerance."""
    timestep = 256
    input_0, input_1 = make_inputs(timestep)

    for device in devices():
        model, reference, error, _, _ = run_accuracy(
            input_0, input_1, device, timestep
        )
        print(f'[{device}]')
        model.accuracy.analyze(reference, verbose=True)
        assert torch.equal(
            model.accuracy.spike_value, model.decoder.spike_value
        ), 'accuracy.spike_value != decoder.spike_value'
        assert error.abs().max() <= 1.0 / math.sqrt(timestep), error.abs().max()
        print(
            f'[{device}] accuracy mae={error.abs().mean().item():.4f}, '
            f'max={error.abs().max().item():.4f}'
        )


def test_reset():
    """Verify resetting accuracy clears its counters and reproduces the decoded stream."""
    timestep = 256
    input_0, input_1 = make_inputs(timestep)

    for device in devices():
        model, _, _, first_value, _ = run_accuracy(
            input_0, input_1, device, timestep
        )
        model.reset()
        assert not model.accuracy.valid
        assert model.accuracy.timestep_cur == 0
        assert torch.equal(
            model.accuracy.spike_count,
            torch.zeros_like(model.accuracy.spike_count),
        )
        _, _, _, second_value, _ = run_accuracy(
            input_0, input_1, device, timestep, model
        )
        assert torch.equal(first_value, second_value), 'reset re-run diverged'


def test_performance():
    """Verify accuracy updates meet the configured runtime bounds across supported devices."""
    timestep = 256
    shape = (100000,)
    torch.manual_seed(0)
    spike_stream = torch.randint(0, 2, (timestep, *shape)).float()
    reference = torch.zeros(shape)

    cpu_runtime = None
    for device in devices():
        metric = accuracy({'polarity': 'bipolar'}).to(device)
        inputs = (spike_stream, reference)

        def run(target_metric, values):
            spikes, _ = values
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
        metric.analyze(reference.to(device))
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
    test_scaled_reference()
    test_fidelity()
    test_reset()
    test_performance()
    print('Test passed.')

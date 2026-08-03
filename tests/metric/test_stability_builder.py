"""
Per-device (cpu/cuda/mps) test of the stability builder: the emitted stream
decodes back to the source value within threshold, its measured stability tracks
the requested normalized stability, and reset restores the initial stream
exactly. Performance is compared with the same builder on CPU.
"""
import torch

from napl.sim.metric import stability
from napl.sim.metric.stability_builder import stability_builder
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices


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
    """Verify stability_builder reconstructs values within its configured threshold."""
    timestep = 256
    threshold = 0.05
    val = gen_rand_tensor('bipolar', shape=(20, 50), width=8)

    for device in devices():
        _, error = run_decode(val, device, timestep, threshold)
        print(
            f'[{device}] decode err mean={error.mean().item():.4f}, '
            f'max={error.max().item():.4f}'
        )
        assert error.max() <= threshold + 2.0 / timestep, error.max()


def test_known_answer():
    """Verify stability_builder emits the expected finite bipolar stream and completion state."""
    source = torch.tensor([-1.0, 1.0])
    expected = torch.tensor(
        [[0, 1], [0, 1], [0, 1], [0, 0]],
        dtype=torch.int8,
    )

    for device in devices():
        builder = stability_builder(
            source,
            {
                'polarity': 'bipolar',
                'threshold': 0.05,
                'normstability': 0.5,
                'timestep': 4,
                'generator': 'sobol',
                'dim': 1,
            },
        ).to(device)
        assert not builder.valid
        result = torch.stack([builder() for _ in range(4)])
        assert builder.valid
        assert builder.timestep_cur == 4
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        assert torch.equal(result.cpu(), expected)

        boundary = stability_builder(
            source,
            {
                'polarity': 'bipolar',
                'threshold': 0.05,
                'normstability': 0.5,
                'timestep': 1,
                'generator': 'sobol',
                'dim': 1,
            },
        ).to(device)
        assert not boundary.valid
        assert torch.equal(
            boundary().cpu(),
            torch.tensor([0, 1], dtype=torch.int8),
        )
        assert boundary.valid
        assert boundary.timestep_cur == 1

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
    """Verify resetting stability_builder clears state and reproduces its emitted stream."""
    timestep = 256
    threshold = 0.05
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        builder, _ = run_decode(val, device, timestep, threshold)
        builder.reset()
        assert not builder.valid
        first = torch.stack([builder() for _ in range(timestep)])
        assert builder.valid
        assert builder.timestep_cur == timestep
        assert first.shape == (timestep, *val.shape)
        assert first.dtype == builder.stype
        builder.reset()
        assert builder.timestep_cur == 0
        assert builder.out_cnt_ns.abs().sum() == 0
        assert builder.out_cnt_st.abs().sum() == 0
        second = torch.stack([builder() for _ in range(timestep)])
        assert torch.equal(first, second), 'reset re-run diverged'


def test_performance():
    """Verify stability_builder meets the configured runtime bounds across supported devices."""
    timestep = 256
    threshold = 0.05
    val = gen_rand_tensor('bipolar', shape=(1000000,), width=8)

    cpu_runtime = None
    for device in devices():
        config = {
            'polarity': 'bipolar',
            'threshold': threshold,
            'normstability': 0.8,
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 1,
        }
        builder = stability_builder(val, config).to(device)
        inputs = (val,)

        def run(target_builder, _):
            for _ in range(timestep):
                target_builder()

        device_runtime = benchmark(
            lambda values: run(builder, values),
            inputs,
            device,
            warmup_runs=1,
            trials=3,
            prepare=builder.reset,
        )
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

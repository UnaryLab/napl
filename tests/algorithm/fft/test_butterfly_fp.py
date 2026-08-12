import torch

from napl.sim.algorithm.fft import butterfly_fp
from napl.sim.base import global_config
from napl.utils._shared_test import benchmark, devices


def test_butterfly_fp_is_napl_module():
    """Verify the non-streaming contract and known answer across devices, including timing."""
    performance_inputs_cpu = tuple(
        torch.linspace(-1, 1, 131072, dtype=global_config.ntype)
        for _ in range(6)
    )

    cpu_runtime = None
    for device in devices():
        operation = butterfly_fp().to(device)
        assert operation.streaming is False
        assert operation.layer == 'algorithm'
        operation.reset()
        assert operation.timestep_cur == 0
        inputs = tuple(
            torch.ones(2, 1, dtype=global_config.ntype, device=device)
            for _ in range(6)
        )
        y0r, y0i, y1r, y1i = operation(*inputs)
        # t = (1*1 - 1*1, 1*1 + 1*1) = (0, 2), so y0 = (1, 3) and y1 = (1, -1).
        assert torch.equal(y0r, inputs[0])
        assert torch.equal(y0i, inputs[0] * 3)
        assert torch.equal(y1r, inputs[0])
        assert torch.equal(y1i, -inputs[0])

        device_runtime = benchmark(
            lambda values: operation(*values),
            performance_inputs_cpu,
            device,
            warmup_runs=1,
            trials=3,
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
    test_butterfly_fp_is_napl_module()
    print('Test passed.')

import math

import torch

from napl.sim.algorithm.fft.butterfly import (
    butterfly_binary,
    butterfly_spike,
)
from napl.sim.base import global_config
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices


def test_butterfly_spike():
    """Verify the streaming complex butterfly matches the decoded FFT reference across devices."""
    torch.manual_seed(0)
    timestep = 256
    codec_config = {
        'polarity': 'bipolar',
        'timestep': timestep,
        'generator': 'sobol',
    }
    width = math.log2(timestep)
    mul_config = {
        'polarity': 'bipolar',
        'timestep': timestep,
        'generator': 'sobol',
    }
    add_config = {
        'polarity': 'bipolar',
        'scale': 3,
        'width': width + 1,
    }
    inputs_cpu = tuple(
        gen_rand_tensor(
            codec_config['polarity'],
            shape=(512, 1),
            width=width,
        ).type(global_config.ntype)
        for _ in range(6)
    )

    cpu_runtime = None
    for device in devices():
        inputs = tuple(value.to(device) for value in inputs_cpu)
        reference_values = butterfly_binary().to(device)(*inputs)
        operation = butterfly_spike(
            codec_config,
            mul_config,
            add_config,
            codec_config,
        ).to(device)

        first = operation(*inputs, timesteps=timestep)
        first = tuple(value.detach().clone() for value in first)
        reference = torch.cat(reference_values, dim=0) / add_config['scale']
        error, _ = operation.accuracy_y.analyze(reference, verbose=True)
        assert error.pow(2).mean().sqrt() < 0.2
        assert operation.timestep_cur == 1

        operation.reset()
        assert operation.timestep_cur == 0
        assert operation._stack_cache is None
        assert operation.encoder_x.timestep_cur == 0
        assert operation.decoder_y.timestep_cur == 0
        assert operation.accuracy_y.timestep_cur == 0
        replay = operation(*inputs, timesteps=timestep)
        assert all(
            torch.equal(before, after)
            for before, after in zip(first, replay)
        )

        device_runtime = benchmark(
            lambda values: operation(*values, timesteps=timestep),
            inputs_cpu,
            device,
            warmup_runs=1,
            trials=3,
            prepare=operation.reset,
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
    test_butterfly_spike()
    print('Test passed.')

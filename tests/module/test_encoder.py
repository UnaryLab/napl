import math

import torch

from napl.sim.base import global_config
from napl.sim.metric import accuracy
from napl.sim.module.encoder import (
    encoder,
    gen_num_seq,
    get_lfsr_seq,
    get_sysrand_seq,
    input_scale,
)
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices


def _run(stream_encoder, stream_accuracy, input, timestep):
    for _ in range(timestep):
        stream_accuracy(stream_encoder(input))


def test_encoder():
    torch.manual_seed(0)
    config = {
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'name': 'spike_accuracy',
        'dim': 1,
    }
    input_cpu = gen_rand_tensor(
        config['polarity'],
        shape=(4096,),
        width=math.log2(config['timestep']),
    ).type(global_config.ntype)

    cpu_runtime = None
    for device in devices():
        stream_encoder = encoder(config).to(device)
        stream_accuracy = accuracy(config).to(device)
        input = input_cpu.to(device)

        _run(stream_encoder, stream_accuracy, input, config['timestep'])
        error, _ = stream_accuracy.analyze(input, verbose=True)
        assert error.pow(2).mean().sqrt() <= 1.0 / math.sqrt(
            config['timestep']
        )
        assert stream_encoder.timestep_cur == config['timestep']
        first = stream_accuracy.spike_value.detach().cpu().clone()

        stream_encoder.reset()
        stream_accuracy.reset()
        assert stream_encoder.timestep_cur == 0
        assert not stream_accuracy.valid
        _run(stream_encoder, stream_accuracy, input, config['timestep'])
        assert torch.equal(first, stream_accuracy.spike_value.cpu())

        device_runtime = benchmark(
            lambda values: _run(
                stream_encoder,
                stream_accuracy,
                values[0],
                config['timestep'],
            ),
            (input_cpu,),
            device,
            warmup_runs=1,
            trials=3,
            prepare=lambda: (
                stream_encoder.reset(),
                stream_accuracy.reset(),
            ),
        )
        if device == 'cpu':
            cpu_runtime = device_runtime
        assert cpu_runtime is not None
        print(
            f'[{device}] device_runtime={device_runtime * 1e3:.1f}ms, '
            f'cpu_runtime={cpu_runtime * 1e3:.1f}ms, '
            f'speedup={cpu_runtime / device_runtime:.2f}x'
        )


def test_encoder_rank2():
    config = {
        'polarity': 'bipolar',
        'timestep': 16,
        'generator': 'sobol',
    }
    input_cpu = torch.tensor([[-0.75, -0.25], [0.25, 0.75]])

    for device in devices():
        stream_encoder = encoder(config).to(device)
        stream_accuracy = accuracy(config).to(device)
        input = input_cpu.to(device)

        _run(stream_encoder, stream_accuracy, input, config['timestep'])
        error, _ = stream_accuracy.analyze(input)

        assert stream_accuracy.spike_value.shape == input.shape
        assert error.shape == input.shape
        assert error.pow(2).mean().sqrt() <= 1.0 / math.sqrt(
            config['timestep']
        )


def test_number_sequences():
    width = 4
    length = 2**width

    torch.manual_seed(0)
    sys_seq = get_sysrand_seq(width)
    assert sys_seq.dtype == torch.float32
    assert torch.equal(
        (sys_seq * length).sort().values,
        torch.arange(length, dtype=sys_seq.dtype),
    )

    lfsr_a = get_lfsr_seq(width=width, seed=1)
    lfsr_b = get_lfsr_seq(width=width, seed=1)
    assert torch.equal(lfsr_a, lfsr_b)
    assert lfsr_a.shape == (length,)
    assert torch.all((lfsr_a >= 0) & (lfsr_a < 1))

    temporal = gen_num_seq({'width': width, 'generator': 'temporal'})
    expected = torch.arange(
        length - 1,
        -1,
        -1,
        dtype=global_config.ntype,
    )
    expected.div_(length)
    assert torch.equal(temporal, expected)


def test_input_scale():
    input = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
    assert torch.equal(input_scale(input), input / 4)


if __name__ == '__main__':
    test_encoder()
    test_encoder_rank2()
    test_number_sequences()
    test_input_scale()
    print('Test passed.')

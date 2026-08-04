import math

import torch

from napl.sim.base import global_config
from napl.sim.metric import accuracy
from napl.sim.operation import encode, decode
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, timer


def test_decode():
    """Verify decoder and accuracy agree on bipolar streams across supported devices."""
    config={
        'polarity': 'bipolar',
        'timestep': 1024,
        'generator': 'sobol',
        'name': 'spike_accuracy',
    }

    input_cpu = gen_rand_tensor(
        config['polarity'],
        shape=(10000,),
        width=math.log2(config['timestep']),
    ).type(global_config.ntype)

    for device in devices():
        spike_encoder = encode(config).to(device)
        spike_accuracy = accuracy(config).to(device)
        spike_decoder = decode(config).to(device)
        assert isinstance(spike_encoder, encode)
        assert spike_encoder.timestep == config['timestep']
        assert spike_encoder.generator == config['generator']
        assert torch.equal(
            spike_decoder.spike_value,
            torch.zeros_like(spike_decoder.spike_count),
        )
        input = input_cpu.to(device)

        with timer(device) as elapsed:
            for _ in range(config['timestep']):
                spike = spike_encoder(input)
                spike_accuracy(spike)
                decoder_result = spike_decoder(spike)

        assert decoder_result is None
        error, _ = spike_accuracy.analyze(input, verbose=True)
        spike_accuracy_value = spike_accuracy.spike_value
        spike_decoder_value = spike_decoder.spike_value
        assert torch.equal(spike_accuracy_value, spike_decoder_value)
        assert error.pow(2).mean().sqrt() <= 1.0 / math.sqrt(config['timestep'])
        print(f'[{device}] time={elapsed.seconds * 1000:.1f}ms')

        spike_encoder.reset()
        spike_decoder.reset()
        spike_accuracy.reset()
        assert spike_encoder.timestep_cur == 0
        assert spike_decoder.timestep_cur == 0
        assert not spike_accuracy.valid
        assert torch.equal(
            spike_decoder.spike_value,
            torch.zeros_like(spike_decoder.spike_count),
        )
    
    print('Test passed.')


def test_decode_rank2():
    """Verify decoder preserves rank-two shape and matches accuracy decoding."""
    config = {
        'polarity': 'bipolar',
        'timestep': 16,
        'generator': 'sobol',
    }
    input_cpu = torch.tensor([[-0.75, -0.25], [0.25, 0.75]])

    for device in devices():
        spike_encoder = encode(config).to(device)
        spike_accuracy = accuracy(config).to(device)
        spike_decoder = decode(config).to(device)
        input = input_cpu.to(device)

        for _ in range(config['timestep']):
            spike = spike_encoder(input)
            spike_accuracy(spike)
            spike_decoder(spike)

        error, _ = spike_accuracy.analyze(input)
        assert spike.shape == input.shape
        assert spike_decoder.spike_value.shape == input.shape
        assert torch.equal(
            spike_accuracy.spike_value,
            spike_decoder.spike_value,
        )
        assert error.pow(2).mean().sqrt() <= 1.0 / math.sqrt(
            config['timestep']
        )


if __name__ == '__main__':
    test_decode()
    test_decode_rank2()

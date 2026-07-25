import math
import time

import torch

from napl.sim.base import global_config
from napl.sim.metric import accuracy
from napl.sim.module import encoder, decoder
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync


def test_decoder():
    """
    Test the decoder with a simple configuration.
    """
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
        spike_encoder = encoder(config).to(device)
        spike_accuracy = accuracy(config).to(device)
        spike_decoder = decoder(config).to(device)
        assert isinstance(spike_encoder, encoder)
        assert spike_encoder.timestep == config['timestep']
        assert spike_encoder.generator == config['generator']
        assert torch.equal(
            spike_decoder.spike_value,
            torch.zeros_like(spike_decoder.spike_count),
        )
        input = input_cpu.to(device)

        sync(device)
        start = time.perf_counter()
        for _ in range(config['timestep']):
            spike = spike_encoder(input)
            spike_accuracy(spike)
            decoder_result = spike_decoder(spike)
        sync(device)
        elapsed = time.perf_counter() - start

        assert decoder_result is None
        error, _ = spike_accuracy.analyze(input, verbose=True)
        spike_accuracy_value = spike_accuracy.spike_value
        spike_decoder_value = spike_decoder.spike_value
        assert torch.equal(spike_accuracy_value, spike_decoder_value)
        assert error.pow(2).mean().sqrt() <= 1.0 / math.sqrt(config['timestep'])
        print(f'[{device}] time={elapsed * 1000:.1f}ms')

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


if __name__ == '__main__':
    test_decoder()

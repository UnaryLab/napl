import math
import time


from napl.base import global_config
from napl.metric import accuracy
from napl.module import encoder
from napl.utils import devices, gen_rand_tensor, sync


def test_encoder():
    """
    Test the encoder with a simple configuration.
    """
    config={
        'polarity': 'bipolar',
        'timestep': 1024,
        'generator': 'sobol',
        'name': 'spike_accuracy',
        'dim': 1
    }

    input_cpu = gen_rand_tensor(
        config['polarity'],
        shape=(10000,),
        width=math.log2(config['timestep']),
    ).type(global_config.ntype)

    for device in devices():
        spike_encoder = encoder(config).to(device)
        spike_accuracy = accuracy(config).to(device)
        assert isinstance(spike_encoder, encoder)
        assert spike_encoder.timestep == config['timestep']
        assert spike_encoder.generator == config['generator']
        input = input_cpu.to(device)

        sync(device)
        start = time.perf_counter()
        for _ in range(config['timestep']):
            spike_accuracy(spike_encoder(input))
        sync(device)
        elapsed = time.perf_counter() - start

        error, _ = spike_accuracy.analyze(input, verbose=True)
        assert error.pow(2).mean().sqrt() <= 1.0 / math.sqrt(config['timestep'])
        print(f'[{device}] time={elapsed * 1000:.1f}ms')

        spike_encoder.reset()
        spike_accuracy.reset()
        assert spike_encoder.timestep_cur == 0
        assert not spike_accuracy.valid
    
    print('Test passed.')


if __name__ == '__main__':
    test_encoder()

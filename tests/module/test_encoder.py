import torch, math

from napl.base import global_config
from napl.metric import accuracy
from napl.module import encoder, decoder
from napl.utils import *


def test_encoder():
    """
    Test the encoder with a simple configuration.
    """
    config={
        'mode': 'bipolar',
        'timestep': 1024,
        'generator': 'sobol',
        'name': 'spike_accuracy',
        'dim': 1
    }

    spike_encoder = encoder(config)
    spike_accuracy = accuracy(config)

    assert isinstance(spike_encoder, encoder), f'Spike encoder should be an instance of encoder class.'
    assert spike_encoder.timestep == config['timestep'], f'Timestep should match.'
    assert spike_encoder.generator == config['generator'], f'Generator should match.'

    input = gen_rand_tensor(config['mode'], shape=(1000,), width=math.log2(config['timestep'])).type(global_config.ntype)

    for _ in range(config['timestep']):
        spike = spike_encoder(input)
        spike_accuracy(spike)


    spike_accuracy.report_error(input, verbose=True)

    spike_encoder.reset()
    spike_accuracy.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_encoder()


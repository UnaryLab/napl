import torch, math

from napl.base import global_config
from napl.metric import accuracy
from napl.module import encoder, decoder
from napl.utils import *


def test_decoder():
    """
    Test the decoder with a simple configuration.
    """
    config={
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'name': 'spike_accuracy',
    }

    spike_encoder = encoder(config)
    spike_accuracy = accuracy(config)
    spike_decoder = decoder(config)
    assert isinstance(spike_encoder, encoder), f'Spike encoder should be an instance of encoder class.'
    assert spike_encoder.timestep == config['timestep'], f'Timestep should match.'
    assert spike_encoder.generator == config['generator'], f'Generator should match.'

    input = gen_rand_tensor(config['mode'], shape=(1000,), bitwidth=math.log2(config['timestep'])).type(global_config.ntype)

    for _ in range(config['timestep']):
        spike = spike_encoder(input)
        spike_accuracy(spike)
        spike_decoder(spike)

    spike_accuracy.report_error(input, verbose=True)

    spike_accuracy_value = spike_accuracy.spike_value
    spike_decoder_value = spike_decoder.spike_value
    assert (spike_accuracy_value == spike_decoder_value).all, f'Spike accuracy value should match spike decoder value.'

    print("Spike decoder test passed.")


if __name__ == "__main__":
    test_decoder()


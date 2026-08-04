import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.module import encoder, decoder
from napl.sim.operation import uni2bi
from napl.sim.metric import accuracy


class napl_uni2bi(napl_base):
    def __init__(self, codec_config1, codec_config2, uni2bi_config):
        super().__init__()
        self.encoder = encoder(codec_config1)
        self.decoder = decoder(codec_config2)
        self.uni2bi = uni2bi(uni2bi_config)
        self.accuracy = accuracy(codec_config2)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.uni2bi(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test uni2bi with a simple configuration.
    """

    codec_config1={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    uni2bi_config={
        'width': 3,
    }
    
    input_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        uni2bi_inst = napl_uni2bi(codec_config1, codec_config2, uni2bi_config).to(device)
        with timer(device) as elapsed:
            uni2bi_inst(input, timesteps=codec_config1['timestep'])

        uni2bi_inst.accuracy.analyze(input, verbose=True)
        assert uni2bi_inst.uni2bi.timestep_cur == codec_config1['timestep']
        uni2bi_inst.reset()
        assert uni2bi_inst.uni2bi.timestep_cur == 0
        print(f'[{device}] time: {elapsed.seconds * 1000:.1f} ms')

    print('Test passed.')


def make_operation(_polarity, _timestep, _device):
    return uni2bi({'width': 3})


def make_values(_polarity):
    return (torch.linspace(0.0, 1.0, 128),)


def make_performance_values(_polarity):
    return (torch.linspace(0.0, 1.0, 131072),)


def analytic_reference(values, _polarity):
    return values[0]


def known_answer_case(_polarity):
    values = torch.tensor([0.0, 0.5, 1.0])
    return (values,), values, 2.0 / math.sqrt(256)


CONFIG = {
    'polarities': ['unipolar'],
    'tolerance_scale': 2.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'input_polarities': ['unipolar'],
    'output_polarity': 'bipolar',
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_uni2bi():
    """Verify uni2bi against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)
    try:
        uni2bi({'width': 2})
    except AssertionError:
        return
    raise AssertionError('uni2bi must reject width 2 because threshold 2 is unreachable')


if __name__ == '__main__':
    test_uni2bi()

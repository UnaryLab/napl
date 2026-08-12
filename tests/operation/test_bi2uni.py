import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import bi2uni, decode, encode
from napl.sim.metric import accuracy


class napl_bi2uni(napl_base):
    def __init__(self, codec_config1, codec_config2, bi2uni_config):
        super().__init__()
        self.encoder = encode(codec_config1)
        self.decoder = decode(codec_config2)
        self.accuracy = accuracy({'polarity': codec_config2['polarity']})
        self.bi2uni = bi2uni(bi2uni_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.bi2uni(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test bi2uni with a simple configuration.
    """

    codec_config1={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    bi2uni_config={
        'width': 2,
    }
    
    # Inputs remain non-negative.
    input_cpu = gen_rand_tensor(
        codec_config2['polarity'],
        shape=(10000,),
        width=math.log2(codec_config1['timestep']),
    ).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        bi2uni_inst = napl_bi2uni(codec_config1, codec_config2, bi2uni_config).to(device)

        with timer(device) as elapsed:
            bi2uni_inst(input, timesteps=codec_config1['timestep'])

        error, _ = bi2uni_inst.accuracy.analyze(input, verbose=True)
        rmse = error.pow(2).mean().sqrt().item()

        assert bi2uni_inst.bi2uni.timestep_cur == codec_config1['timestep']
        bi2uni_inst.reset()
        assert bi2uni_inst.bi2uni.timestep_cur == 0
        print(f'[{device}] rmse={rmse:.4f}, time={elapsed.seconds:.3f}s')

    print('Test passed.')


def make_operation(_polarity, _timestep, _device):
    return bi2uni({'width': 2})


def make_values(_polarity):
    return (torch.linspace(0.0, 1.0, 128),)


def make_random_perf_values(_polarity):
    return (torch.linspace(0.0, 1.0, 131072),)


def analytic_reference(values, _polarity):
    return values[0]


def known_answer_case(_polarity):
    values = torch.tensor([0.0, 0.5, 1.0])
    return (values,), values


CONFIG = {
    'polarities': ['bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'input_polarities': ['bipolar'],
    'output_polarity': 'unipolar',
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_bi2uni():
    """Verify bi2uni against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)
    try:
        bi2uni({'width': 1})
    except AssertionError:
        return
    raise AssertionError('bi2uni must reject width 1 because threshold 1 is unreachable')


if __name__ == '__main__':
    test_bi2uni()

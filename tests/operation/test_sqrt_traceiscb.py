import math
import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import sqrt_traceiscb
from napl.sim.metric import accuracy


class napl_sqrt_traceiscb(napl_base):
    def __init__(self, codec_config, sqrt_traceiscb_config):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.sqrt_traceiscb = sqrt_traceiscb(sqrt_traceiscb_config)
        self.accuracy = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.sqrt_traceiscb(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test sqrt_traceiscb with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    sqrt_traceiscb_config={
        'polarity': 'bipolar',
    }
    
    input_cpu = gen_rand_tensor('unipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        sqrt_traceiscb_inst = napl_sqrt_traceiscb(codec_config, sqrt_traceiscb_config).to(device)
        sync(device)
        start = time.perf_counter()
        sqrt_traceiscb_inst(input, timesteps=codec_config['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = torch.sqrt(input)
        sqrt_traceiscb_inst.accuracy.analyze(r_value, verbose=True)
        assert sqrt_traceiscb_inst.sqrt_traceiscb.timestep_cur == codec_config['timestep']
        sqrt_traceiscb_inst.reset()
        assert sqrt_traceiscb_inst.sqrt_traceiscb.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')

    print('Test passed.')


def make_operation(polarity, _timestep, _device):
    return sqrt_traceiscb({'polarity': polarity})


def make_values(_polarity):
    return (torch.linspace(0.0, 1.0, 128),)


def analytic_reference(values, _polarity):
    return torch.sqrt(values[0])


def known_answer_case(_polarity):
    values = torch.tensor([0.25, 1.0])
    return (values,), torch.sqrt(values), 0.35


CONFIG = {
    'polarities': ['bipolar'],
    'tolerance_scale': 5.5,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_sqrt_traceiscb():
    """Verify sqrt_traceiscb against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


def test_sqrt_traceiscb_matches_unarysim_trace():
    """Verify sqrt_traceiscb reproduces UnarySim's startup trace for both polarities."""
    expected = torch.tensor([0, 0, 0, 1], dtype=global_config.stype)

    for polarity in ('unipolar', 'bipolar'):
        operation = sqrt_traceiscb({'polarity': polarity})
        output = torch.stack([
            operation(torch.zeros((2, 3), dtype=global_config.stype))
            for _ in range(4)
        ])

        assert torch.equal(output, expected.view(-1, 1, 1).expand_as(output))


if __name__ == '__main__':
    test_sqrt_traceiscb()
    test_sqrt_traceiscb_matches_unarysim_trace()

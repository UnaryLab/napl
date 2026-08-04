import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.module import encoder, decoder
from napl.sim.operation import sigmoid_hard
from napl.sim.metric import accuracy


class napl_sigmoid_hard(napl_base):
    def __init__(self, codec_config, sigmoid_hard_config):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.sigmoid_hard = sigmoid_hard(sigmoid_hard_config)
        self.accuracy = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.sigmoid_hard(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test sigmoid_hard with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    sigmoid_hard_config={
        'polarity': 'bipolar',
    }
    
    input_cpu = gen_rand_tensor('bipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        sigmoid_hard_inst = napl_sigmoid_hard(codec_config, sigmoid_hard_config).to(device)
        with timer(device) as elapsed:
            sigmoid_hard_inst(input, timesteps=codec_config['timestep'])

        r_value = torch.nn.Hardsigmoid()(input * 3)
        sigmoid_hard_inst.accuracy.analyze(r_value, verbose=True)
        assert sigmoid_hard_inst.sigmoid_hard.timestep_cur == codec_config['timestep']
        sigmoid_hard_inst.reset()
        assert sigmoid_hard_inst.sigmoid_hard.timestep_cur == 0
        print(f'[{device}] time: {elapsed.seconds * 1000:.1f} ms')

    print('Test passed.')


def make_operation(polarity, _timestep, _device):
    return sigmoid_hard({'polarity': polarity})


def make_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(lo, hi, 128),)


def make_performance_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(lo, hi, 131072),)


def analytic_reference(values, _polarity):
    return torch.nn.functional.hardsigmoid(values[0] * 3)


def known_answer_case(polarity):
    values = torch.tensor([0.0, 0.5, 1.0]) if polarity == 'unipolar' else torch.tensor([-1.0, 0.0, 1.0])
    expected = torch.nn.functional.hardsigmoid(values * 3)
    return (values,), expected, 3.0 / math.sqrt(256)


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 3.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_sigmoid_hard():
    """Verify sigmoid_hard for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


def test_sigmoid_hard_uses_inclusive_carry_sequence():
    """Verify sigmoid_hard uses add_any's inclusive carry sequence for both polarities."""
    input_values = [1, 1, 0]
    expected = torch.tensor([1, 1, 0], dtype=global_config.stype)

    for polarity in ('unipolar', 'bipolar'):
        operation = sigmoid_hard({'polarity': polarity})
        output = torch.stack([
            operation(torch.full((2, 3), value, dtype=global_config.stype))
            for value in input_values
        ])
        assert torch.equal(output, expected.view(-1, 1, 1).expand_as(output))


if __name__ == '__main__':
    test_sigmoid_hard()
    test_sigmoid_hard_uses_inclusive_carry_sequence()

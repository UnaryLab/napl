import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import encode, decode
from napl.sim.operation import square_dff
from napl.sim.metric import accuracy


class napl_square_dff(napl_base):
    def __init__(self, codec_config, square_dff_config):
        super().__init__()
        self.encoder = encode(codec_config)
        self.decoder = decode(codec_config)
        self.square_dff = square_dff(square_dff_config)
        self.accuracy = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.square_dff(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test square_dff with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    square_dff_config={
        'polarity': 'bipolar',
        'depth': 1
    }
    
    input_cpu = gen_rand_tensor(codec_config['polarity'], shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        square_dff_inst = napl_square_dff(codec_config, square_dff_config).to(device)
        with timer(device) as elapsed:
            square_dff_inst(input, timesteps=codec_config['timestep'])

        r_value = input * input
        error, _ = square_dff_inst.accuracy.analyze(r_value, verbose=True)
        rmse = error.pow(2).mean().sqrt()
        assert rmse <= CONFIG['tolerance_scale'] / math.sqrt(codec_config['timestep']), rmse
        assert square_dff_inst.square_dff.timestep_cur == codec_config['timestep']
        square_dff_inst.reset()
        assert square_dff_inst.square_dff.timestep_cur == 0
        print(f'[{device}] time: {elapsed.seconds * 1000:.1f} ms')

    print('Test passed.')


def make_operation(polarity, _timestep, _device):
    return square_dff({'polarity': polarity, 'depth': 1})


def make_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(lo, hi, 128),)


def make_performance_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(lo, hi, 131072),)


def analytic_reference(values, _polarity):
    return values[0].square()


def known_answer_case(polarity):
    values = torch.tensor([0.0, 0.5, 1.0]) if polarity == 'unipolar' else torch.tensor([-1.0, 0.0, 1.0])
    return (values,), values.square(), 0.35


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 5.5,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_square_dff():
    """Verify square_dff for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_square_dff()

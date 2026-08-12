import torch
import math

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import decode, encode, sqrt_gaines
from napl.sim.metric import accuracy


class napl_sqrt_gaines(napl_base):
    def __init__(self, codec_config, sqrt_gaines_config):
        super().__init__()
        self.encoder = encode(codec_config)
        self.decoder = decode(codec_config)
        self.sqrt_gaines = sqrt_gaines(sqrt_gaines_config)
        self.accuracy = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.sqrt_gaines(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def run_sqrt_gaines(polarity, device):
    codec_config={
        'polarity': polarity,
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    sqrt_gaines_config={
        'polarity': polarity,
        'width': 5,
        'generator': 'sobol',
        'dim': 4,
    }

    # Square root inputs are non-negative.
    input = gen_rand_tensor('unipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    sqrt_gaines_inst = napl_sqrt_gaines(codec_config, sqrt_gaines_config).to(device)

    with timer(device) as elapsed:
        sqrt_gaines_inst(input, timesteps=codec_config['timestep'])

    r_value = torch.sqrt(input)

    sqrt_gaines_inst.accuracy.analyze(r_value, verbose=True)

    rmse = (sqrt_gaines_inst.decoder.spike_value - r_value).pow(2).mean().sqrt().item()

    assert sqrt_gaines_inst.sqrt_gaines.timestep_cur == codec_config['timestep']
    sqrt_gaines_inst.reset()
    assert sqrt_gaines_inst.sqrt_gaines.timestep_cur == 0

    print(f'{polarity} on {device}: RMSE {rmse:.4f}, {elapsed.seconds:.2f} s')


def _kernel_specific_checks():
    """
    Test sqrt_gaines on every available device, for both polarities.
    """
    for device in devices():
        for polarity in ['unipolar', 'bipolar']:
            run_sqrt_gaines(polarity, device)

    print('Test passed.')


def make_operation(polarity, _timestep, _device):
    return sqrt_gaines({
        'polarity': polarity,
        'width': 5,
        'generator': 'sobol',
        'dim': 4,
    })


def make_values(_polarity):
    # The Gaines counter floor is near one third, so fidelity inputs start at 0.15.
    return (torch.linspace(0.15, 1.0, 128),)


def make_random_perf_values(_polarity):
    return (torch.linspace(0.0, 1.0, 131072),)


def analytic_reference(values, _polarity):
    return torch.sqrt(values[0])


def known_answer_case(_polarity):
    values = torch.tensor([0.25, 1.0])
    return (values,), torch.sqrt(values)


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_sqrt_gaines():
    """Verify sqrt_gaines for both polarities on the non-negative square-root domain."""
    # The kernel holds its own comparison-reference encoder.
    assert make_operation('bipolar', 256, 'cpu').internal_encode is True
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_sqrt_gaines()

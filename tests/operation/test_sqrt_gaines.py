import torch
import math
import time

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation.sqrt_gaines import sqrt_gaines
from napl.sim.metric import accuracy


class napl_sqrt_gaines(napl_base):
    def __init__(self, codec_config, sqrt_gaines_config):
        super().__init__()
        # set up encoder, decoder, sqrt, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.sqrt_gaines = sqrt_gaines(sqrt_gaines_config)
        self.accuracy = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
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

    # Generate random inputs, ensure non-negative numbers for sqrt
    input = gen_rand_tensor('unipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype).to(device)

    # generate the napl_sqrt_gaines instance
    sqrt_gaines_inst = napl_sqrt_gaines(codec_config, sqrt_gaines_config).to(device)

    sync(device)
    start_time = time.perf_counter()
    sqrt_gaines_inst(input, timesteps=codec_config['timestep'])
    sync(device)
    elapsed = time.perf_counter() - start_time

    # calculate the reference output
    r_value = torch.sqrt(input)

    # report the error
    sqrt_gaines_inst.accuracy.analyze(r_value, verbose=True)

    rmse = (sqrt_gaines_inst.decoder.spike_value - r_value).pow(2).mean().sqrt().item()
    # Gaines sqrt uses a feedback counter and is biased near zero input, so accuracy is
    # looser than the SC bound 1/sqrt(N); bound set with headroom over measured RMSE
    # (~0.09 unipolar, ~0.10 bipolar at T=256, matching UnarySim's GainesSqrt error profile)
    bound = 0.15
    assert rmse < bound, f'RMSE {rmse} exceeds bound {bound} for {polarity} on {device}'

    assert sqrt_gaines_inst.sqrt_gaines.timestep_cur == codec_config['timestep']
    sqrt_gaines_inst.reset()
    assert sqrt_gaines_inst.sqrt_gaines.timestep_cur == 0

    print(f'{polarity} on {device}: RMSE {rmse:.4f}, {elapsed:.2f} s')


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
    return (torch.linspace(0.05, 1.0, 128),)


def analytic_reference(values, _polarity):
    return torch.sqrt(values[0])


def known_answer_case(_polarity):
    values = torch.tensor([0.25, 1.0])
    return (values,), torch.sqrt(values), 0.15


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 2.4,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_sqrt_gaines():
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_sqrt_gaines()

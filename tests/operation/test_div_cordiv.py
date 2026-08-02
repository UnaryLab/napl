import math
import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import div_cordiv
from napl.sim.metric import accuracy


class napl_div_cordiv(napl_base):
    def __init__(self, codec_config1, codec_config2, div_cordiv_config):
        super().__init__()
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder = decoder(codec_config1)
        self.accuracy = accuracy({'polarity': codec_config1['polarity']})
        self.div_cordiv = div_cordiv(div_cordiv_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.div_cordiv(i_spike0, i_spike1)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test div_cordiv with a simple configuration.
    """

    codec_config1={
        'polarity': 'unipolar',
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
    div_cordiv_config={
        'depth' : 2, 
        'generator' : 'Sobol',
    }

    # Matching codec dimensions provide the correlated spikes required by div_cordiv.
    input_0 = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1 = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)
    input_mask = input_0 < input_1
    input_0_new = torch.where(input_mask, input_0, input_1)
    input_1_new = torch.where(~input_mask, input_0, input_1)
    # Division requires a nonzero divisor.
    input_1_new = torch.where(input_1_new==0, 1, input_1_new)
    input_0 = input_0_new
    input_1 = input_1_new

    for device in devices():
        dividend = input_0.to(device)
        divisor = input_1.to(device)
        div_cordiv_inst = napl_div_cordiv(codec_config1, codec_config2, div_cordiv_config).to(device)

        sync(device)
        start = time.perf_counter()
        div_cordiv_inst(dividend, divisor, timesteps=codec_config1['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = dividend / divisor
        error, _ = div_cordiv_inst.accuracy.analyze(r_value, verbose=True)
        rmse = error.pow(2).mean().sqrt().item()
        assert rmse < 0.2, f'[{device}] rmse={rmse:.4f}'

        assert div_cordiv_inst.div_cordiv.timestep_cur == codec_config1['timestep']
        div_cordiv_inst.reset()
        assert div_cordiv_inst.div_cordiv.timestep_cur == 0
        print(f'[{device}] rmse={rmse:.4f}, time={elapsed:.3f}s')
    
    print('Test passed.')


def make_operation(_polarity, _timestep, _device):
    return div_cordiv({'depth': 2, 'generator': 'Sobol'})


def make_values(_polarity):
    quotient = torch.linspace(0.1, 0.9, 128)
    divisor = torch.full_like(quotient, 0.875)
    return quotient * divisor, divisor


def analytic_reference(values, _polarity):
    return values[0] / values[1]


def known_answer_case(_polarity):
    values = (torch.tensor([0.25]), torch.tensor([0.5]))
    return values, torch.tensor([0.5]), 0.2


CONFIG = {
    'polarities': ['unipolar'],
    'tolerance_scale': 3.2,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [1, 1],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_div_cordiv():
    """Verify div_cordiv against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


def test_div_cordiv_matches_unarysim_history_order():
    """Verify div_cordiv reproduces UnarySim's quotient history update order."""
    dividends = [0, 1, 0, 1, 1, 0]
    divisors = [0, 0, 1, 0, 1, 0]
    expected = torch.tensor([0, 1, 0, 0, 1, 1], dtype=global_config.stype)
    operation = div_cordiv({'depth': 2, 'generator': 'Sobol'})

    output = torch.stack([
        operation(
            torch.full((2, 3), dividend, dtype=global_config.stype),
            torch.full((2, 3), divisor, dtype=global_config.stype),
        )
        for dividend, divisor in zip(dividends, divisors)
    ])

    assert torch.equal(output, expected.view(-1, 1, 1).expand_as(output))


if __name__ == '__main__':
    test_div_cordiv()
    test_div_cordiv_matches_unarysim_history_order()

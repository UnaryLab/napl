import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.module import encoder, decoder
from napl.sim.operation import div_iscb
from napl.sim.metric import accuracy


class napl_div_iscb(napl_base):
    def __init__(self, codec_config1, codec_config2, div_iscb_config):
        super().__init__()
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder = decoder(codec_config1)
        self.accuracy = accuracy({'polarity': codec_config1['polarity']})
        self.div_iscb = div_iscb(div_iscb_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.div_iscb(i_spike0, i_spike1)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test div_iscb with a simple configuration.
    """

    polarity = 'bipolar'
    codec_config1={
        'polarity': polarity,
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': polarity,
        'timestep': 256,
        'generator': 'sobol',
        'dim': 2,
    }
    div_iscb_config={
        'polarity': polarity,
    }

    input_0 = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1 = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)
    input_mask = torch.abs(input_0) < torch.abs(input_1)
    input_0_new = torch.where(input_mask, input_0, input_1)
    input_1_new = torch.where(~input_mask, input_0, input_1)
    # Division requires a nonzero divisor.
    input_1_new = torch.where(input_1_new==0, 1, input_1_new)
    input_0 = input_0_new
    input_1 = input_1_new

    for device in devices():
        dividend = input_0.to(device)
        divisor = input_1.to(device)
        div_iscb_inst = napl_div_iscb(codec_config1, codec_config2, div_iscb_config).to(device)

        with timer(device) as elapsed:
            div_iscb_inst(dividend, divisor, timesteps=codec_config1['timestep'])

        r_value = dividend / divisor
        error, _ = div_iscb_inst.accuracy.analyze(r_value, verbose=True)
        rmse = error.pow(2).mean().sqrt().item()
        assert rmse < 0.25, f'[{device}] rmse={rmse:.4f}'

        assert div_iscb_inst.div_iscb.timestep_cur == codec_config1['timestep']
        div_iscb_inst.reset()
        assert div_iscb_inst.div_iscb.timestep_cur == 0
        print(f'[{device}] rmse={rmse:.4f}, time={elapsed.seconds:.3f}s')
    
    print('Test passed.')


def make_operation(polarity, _timestep, _device):
    return div_iscb({'polarity': polarity})


def make_values(polarity):
    if polarity == 'unipolar':
        quotient = torch.linspace(0.0, 1.0, 128)
        divisor = torch.linspace(0.25, 1.0, 128)
    else:
        quotient = torch.linspace(-1.0, 1.0, 128)
        magnitude = torch.linspace(0.25, 1.0, 64)
        divisor = torch.cat((magnitude, -magnitude))
    return quotient * divisor, divisor


def make_performance_values(polarity):
    if polarity == 'unipolar':
        quotient = torch.linspace(0.0, 1.0, 131072)
        divisor = torch.linspace(0.25, 1.0, 131072)
    else:
        quotient = torch.linspace(-1.0, 1.0, 131072)
        magnitude = torch.linspace(0.25, 1.0, 65536)
        divisor = torch.cat((magnitude, -magnitude))
    return quotient * divisor, divisor


def analytic_reference(values, _polarity):
    return values[0] / values[1]


def known_answer_case(polarity):
    if polarity == 'unipolar':
        values = (torch.tensor([0.0, 1.0]), torch.tensor([0.25, 1.0]))
        return values, torch.tensor([0.0, 1.0]), 0.25
    values = (torch.tensor([-1.0, 1.0]), torch.tensor([1.0, -1.0]))
    return values, torch.tensor([-1.0, -1.0]), 0.25


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 4.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_div_iscb():
    """Verify div_iscb for both polarities with legal quotients and nonzero divisors."""
    streaming_suite(CONFIG)


def test_div_iscb_matches_unarysim_history():
    """Verify div_iscb reproduces UnarySim's initial quotient history by polarity."""
    cases = {
        'unipolar': ([0, 0], [0, 0], [0, 1]),
        'bipolar': ([0, 0], [1, 0], [0, 0]),
    }

    for polarity, (dividends, divisors, expected_values) in cases.items():
        operation = div_iscb({'polarity': polarity})
        output = torch.stack([
            operation(
                torch.full((2, 3), dividend, dtype=global_config.stype),
                torch.full((2, 3), divisor, dtype=global_config.stype),
            )
            for dividend, divisor in zip(dividends, divisors)
        ])
        expected = torch.tensor(expected_values, dtype=global_config.stype)
        assert torch.equal(output, expected.view(-1, 1, 1).expand_as(output))


if __name__ == '__main__':
    test_div_iscb()
    test_div_iscb_matches_unarysim_history()

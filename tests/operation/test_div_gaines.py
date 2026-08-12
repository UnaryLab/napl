import torch
import math

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import decode, div_gaines, encode
from napl.sim.metric import accuracy


class napl_div_gaines(napl_base):
    def __init__(self, codec_config1, codec_config2, div_gaines_config):
        super().__init__()
        self.encoder0 = encode(codec_config1)
        self.encoder1 = encode(codec_config2)
        self.decoder = decode(codec_config1)
        self.accuracy = accuracy({'polarity': codec_config1['polarity']})
        self.div_gaines = div_gaines(div_gaines_config)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.div_gaines(i_spike0, i_spike1)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def run_div_gaines(device, polarity, quotient_cpu, divisor_cpu):
    timestep = 256

    codec_config1={
        'polarity': polarity,
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': polarity,
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 2,
    }
    div_gaines_config={
        'polarity': polarity,
        'width': 5,
        'generator': 'sobol',
        'dim': 3,
    }

    # Construct an in-range quotient and keep |divisor| >= 0.5 for convergence.
    quotient = quotient_cpu.to(device)
    divisor = divisor_cpu.to(device)
    dividend = quotient * divisor

    div_gaines_inst = napl_div_gaines(codec_config1, codec_config2, div_gaines_config).to(device)

    with timer(device) as elapsed:
        div_gaines_inst(dividend, divisor, timesteps=timestep)

    error, _ = div_gaines_inst.accuracy.analyze(quotient, verbose=True)
    rmse = error.pow(2).mean().sqrt().item()
    print(f'div_gaines [{device}] [{polarity}] rmse={rmse:.4f} time={elapsed.seconds:.3f}s')

    assert div_gaines_inst.div_gaines.timestep_cur == timestep
    div_gaines_inst.reset()
    assert div_gaines_inst.div_gaines.timestep_cur == 0
    assert div_gaines_inst.div_gaines.reference_encode.timestep_cur == 0


def _kernel_specific_checks():
    """
    Test div_gaines on every available device, both polarities.
    """
    timestep = 256
    cases = {}
    for polarity in ['unipolar', 'bipolar']:
        quotient = gen_rand_tensor(
            polarity,
            shape=(10000,),
            width=math.log2(timestep),
        ).type(global_config.ntype)
        divisor = gen_rand_tensor(
            'unipolar',
            shape=(10000,),
            width=math.log2(timestep),
        ).type(global_config.ntype).div(2).add(0.5)
        if polarity == 'bipolar':
            divisor *= torch.where(torch.rand_like(divisor) < 0.5, -1.0, 1.0)
        cases[polarity] = (quotient, divisor)

    for device in devices():
        for polarity in ['unipolar', 'bipolar']:
            run_div_gaines(device, polarity, *cases[polarity])

    print('Test passed.')


def make_operation(polarity, _timestep, _device):
    return div_gaines({
        'polarity': polarity,
        'width': 5,
        'generator': 'sobol',
        'dim': 3,
    })


def make_values(polarity):
    if polarity == 'unipolar':
        quotient = torch.linspace(0.0, 1.0, 128)
        # The low-divisor end is settling-limited: divisor d needs about 1/d times the stream length.
        divisor = torch.linspace(0.25, 1.0, 128)
    else:
        quotient = torch.linspace(-1.0, 1.0, 128)
        magnitude = torch.linspace(0.25, 1.0, 64)
        divisor = torch.cat((magnitude, -magnitude))
    return quotient * divisor, divisor


def make_random_perf_values(polarity):
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
        # The Gaines divider needs about 1/d times the stream length to settle for divisor d.
        values = (torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0]))
        expected = torch.tensor([0.0, 1.0])
    else:
        values = (torch.tensor([-1.0, 1.0]), torch.tensor([1.0, -1.0]))
        expected = torch.tensor([-1.0, -1.0])
    return values, expected


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


def test_div_gaines():
    """Verify div_gaines with quotient in its legal range and nonzero divisors."""
    # The kernel holds its own comparison-reference encoder.
    assert make_operation('bipolar', 256, 'cpu').internal_encode is True
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_div_gaines()

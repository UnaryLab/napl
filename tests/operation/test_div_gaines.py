import torch
import math
import time

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, sync
from napl.sim.module import encoder, decoder
# div_gaines is not exported from operation/__init__.py.
from napl.sim.operation.div_gaines import div_gaines
from napl.sim.metric import accuracy


class napl_div_gaines(napl_base):
    def __init__(self, codec_config1, codec_config2, div_gaines_config):
        super().__init__()
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder = decoder(codec_config1)
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
        'depth': 5,
        'generator': 'sobol',
        'dim': 3,
    }

    # Construct an in-range quotient and keep |divisor| >= 0.5 for convergence.
    quotient = quotient_cpu.to(device)
    divisor = divisor_cpu.to(device)
    dividend = quotient * divisor

    div_gaines_inst = napl_div_gaines(codec_config1, codec_config2, div_gaines_config).to(device)

    sync(device)
    start = time.perf_counter()
    div_gaines_inst(dividend, divisor, timesteps=timestep)
    sync(device)
    elapsed = time.perf_counter() - start

    error, _ = div_gaines_inst.accuracy.analyze(quotient, verbose=True)
    rmse = error.pow(2).mean().sqrt().item()
    print(f'div_gaines [{device}] [{polarity}] rmse={rmse:.4f} time={elapsed:.3f}s')

    # Feedback makes this bound looser than the open-loop SC bound.
    assert rmse < 0.2, f'rmse {rmse} out of bound on {device} ({polarity})'

    assert div_gaines_inst.div_gaines.timestep_cur == timestep
    div_gaines_inst.reset()
    assert div_gaines_inst.div_gaines.timestep_cur == 0
    assert div_gaines_inst.div_gaines.idx == 0


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
        'depth': 5,
        'generator': 'sobol',
        'dim': 3,
    })


def make_values(polarity):
    quotient = torch.linspace(
        -0.75 if polarity == 'bipolar' else 0.05, 0.75, 128
    )
    divisor = torch.full_like(quotient, 0.75)
    if polarity == 'bipolar':
        divisor[::2] = -0.75
    return quotient * divisor, divisor


def analytic_reference(values, _polarity):
    return values[0] / values[1]


def known_answer_case(polarity):
    if polarity == 'unipolar':
        values = (torch.tensor([0.25]), torch.tensor([0.5]))
        expected = torch.tensor([0.5])
    else:
        values = (torch.tensor([-0.25, 0.25]), torch.tensor([0.5, -0.5]))
        expected = torch.tensor([-0.5, -0.5])
    return values, expected, 0.2


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 3.2,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_div_gaines():
    """Verify div_gaines against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_div_gaines()

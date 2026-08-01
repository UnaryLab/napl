import math
import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import add_any
from napl.sim.metric import accuracy


class napl_add_any(napl_base):
    def __init__(self, codec_config, add_any_config):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.accuracy = accuracy({'polarity': codec_config['polarity']})
        self.add_any = add_any(add_any_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.add_any(i_spike, dim=-1)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test add_any with a simple configuration.
    """
    
    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    add_any_config={
        'polarity': 'bipolar',
        'scale': 128,
        'width': 20,
    }

    input_cpu = gen_rand_tensor(
        codec_config['polarity'],
        shape=(10000, add_any_config['scale']),
        width=math.log2(codec_config['timestep']),
    ).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        add_any_inst = napl_add_any(codec_config, add_any_config).to(device)

        sync(device)
        start = time.perf_counter()
        add_any_inst(input, timesteps=codec_config['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = torch.sum(input, dim=-1) / add_any_config['scale']
        error, _ = add_any_inst.accuracy.analyze(r_value, verbose=True)
        rmse = error.pow(2).mean().sqrt().item()
        bound = 2.0 / math.sqrt(codec_config['timestep'])
        assert rmse < bound, f'[{device}] rmse={rmse:.4f}, bound={bound:.4f}'

        assert add_any_inst.add_any.timestep_cur == codec_config['timestep']
        add_any_inst.reset()
        assert add_any_inst.add_any.timestep_cur == 0
        print(f'[{device}] rmse={rmse:.4f}, time={elapsed:.3f}s')
    
    print('Test passed.')


def make_operation(_polarity, _timestep, _device):
    return add_any({'polarity': 'bipolar', 'scale': 8, 'width': 20})


def make_values(_polarity):
    return (torch.linspace(-0.75, 0.75, 512).reshape(64, 8),)


def analytic_reference(values, _polarity):
    return values[0].mean(dim=-1)


def known_answer_case(_polarity):
    values = torch.full((8, 8), 0.25)
    return (values,), values.mean(dim=-1), 2.0 / math.sqrt(256)


CONFIG = {
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'polarities': ['bipolar'],
    'timesteps': 256,
    'tolerance_scale': 2.0,
    'apply_operation': lambda operation, spikes: operation(spikes[0], dim=-1),
    'extra_checks': _kernel_specific_checks,
}


def test_add_any():
    streaming_suite(CONFIG)


def test_add_any_uses_unarysim_strict_carry_threshold():
    input = torch.ones((2, 2), dtype=global_config.stype)

    for polarity in ('unipolar', 'bipolar'):
        operation = add_any({'polarity': polarity, 'scale': 2, 'width': 8})
        output = operation(input, dim=-1)
        assert torch.equal(output, torch.zeros(2, dtype=global_config.stype))


if __name__ == '__main__':
    test_add_any()
    test_add_any_uses_unarysim_strict_carry_threshold()

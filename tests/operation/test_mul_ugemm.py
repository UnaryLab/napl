import math

import torch

from napl.sim.base import global_config
from napl.sim.operation import mul_ugemm
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256


def make_operation(polarity, timestep, device):
    return mul_ugemm({
        'polarity': polarity,
        'timestep': timestep,
        'generator': 'sobol',
    })


def apply_operation(operation, spikes, values):
    return operation(spikes[0], values[1])


def make_values(polarity):
    return (
        gen_rand_tensor(
            polarity, shape=(10000,), width=math.log2(TIMESTEPS)
        ).type(global_config.ntype),
        gen_rand_tensor(
            polarity, shape=(10000,), width=math.log2(TIMESTEPS)
        ).type(global_config.ntype),
    )


def make_performance_values(polarity):
    low = -1.0 if polarity == 'bipolar' else 0.0
    values = torch.linspace(low, 1.0, 131072, dtype=global_config.ntype)
    return values, values.roll(17)


def analytic_reference(values, polarity):
    return values[0] * values[1]


def known_answer_case(polarity):
    if polarity == 'unipolar':
        values = (
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
        )
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
    else:
        values = (
            torch.tensor([-1.0, -1.0, 1.0, 1.0]),
            torch.tensor([-1.0, 1.0, -1.0, 1.0]),
        )
        expected = torch.tensor([1.0, -1.0, -1.0, 1.0])
    return values, expected, 0.0


def check_rank2():
    """Verify mul_ugemm preserves rank-two spike tensors when multiplied by one."""
    config = {
        'polarity': 'bipolar',
        'timestep': 4,
        'generator': 'sobol',
    }
    input_0_cpu = torch.tensor(
        [[0, 1, 0], [1, 0, 1]],
        dtype=global_config.stype,
    )
    input_1_cpu = torch.ones((2, 3), dtype=global_config.ntype)

    for device in devices():
        operation = mul_ugemm(config).to(device)
        result = operation(input_0_cpu.to(device), input_1_cpu.to(device))

        assert result.ndim == 2
        assert torch.equal(result.cpu(), input_0_cpu)


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 1.0,
    'make_operation': make_operation,
    'apply_operation': apply_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'extra_checks': check_rank2,
    'timesteps': TIMESTEPS,
}


def test_mul_ugemm():
    """Verify mul_ugemm against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_mul_ugemm()

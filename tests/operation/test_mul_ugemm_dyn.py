import math

import torch

from napl.sim.base import global_config
from napl.sim.operation import mul_ugemm_dyn
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import streaming_suite


TIMESTEPS = 256


def make_operation(polarity, timestep, device):
    return mul_ugemm_dyn(
        {
            'polarity': polarity,
            'width': 4,
            'generator': 'sobol',
            'dim': 1,
        }
    )


def make_values(polarity):
    return (
        gen_rand_tensor(
            polarity, shape=(4096,), width=math.log2(TIMESTEPS)
        ).type(global_config.ntype),
        gen_rand_tensor(
            polarity, shape=(4096,), width=math.log2(TIMESTEPS)
        ).type(global_config.ntype),
    )


def make_random_perf_values(polarity):
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
    return values, expected


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': TIMESTEPS,
}


def test_mul_ugemm_dyn():
    """Verify mul_ugemm_dyn against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_mul_ugemm_dyn()

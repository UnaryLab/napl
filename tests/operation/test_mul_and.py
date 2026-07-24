import math

import torch

from napl.sim.base import global_config
from napl.sim.operation import mul_and
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import streaming_suite


TIMESTEPS = 256


def make_operation(polarity, timestep, device):
    return mul_and({'polarity': polarity})


def make_values(polarity):
    return (
        gen_rand_tensor(
            polarity, shape=(10000,), width=math.log2(TIMESTEPS)
        ).type(global_config.ntype),
        gen_rand_tensor(
            polarity, shape=(10000,), width=math.log2(TIMESTEPS)
        ).type(global_config.ntype),
    )


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


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 3.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': TIMESTEPS,
}


def test_mul_and():
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_mul_and()

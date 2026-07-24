import math

import torch

from napl.sim.base import global_config
from napl.sim.operation import mul_gaines
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256
BOUND_SCALE = 3.0


def make_operation(polarity, timestep, device):
    return mul_gaines({'polarity': polarity})


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
    'tolerance_scale': BOUND_SCALE,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': TIMESTEPS,
}


def test_mul_gaines():
    streaming_suite(CONFIG)


def test_mul_gaines_truth_table():
    a = torch.tensor([0, 0, 1, 1], dtype=global_config.stype)
    b = torch.tensor([0, 1, 0, 1], dtype=global_config.stype)
    for device in devices():
        for polarity, expected in (
            ('unipolar', [0, 0, 0, 1]),
            ('bipolar', [1, 0, 0, 1]),
        ):
            operation = mul_gaines({'polarity': polarity}).to(device)
            result = operation(a.to(device), b.to(device)).cpu()
            assert torch.equal(
                result,
                torch.tensor(expected, dtype=global_config.stype),
            )
            operation.reset()
            assert operation.timestep_cur == 0


if __name__ == '__main__':
    test_mul_gaines()
    test_mul_gaines_truth_table()

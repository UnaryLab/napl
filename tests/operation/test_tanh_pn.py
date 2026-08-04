import math

import torch

from napl.sim.base import global_config
from napl.sim.operation import tanh_pn
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import streaming_suite


TIMESTEPS = 256
DEPTH = 3
GAIN = 2 ** (DEPTH - 1)


def make_operation(polarity, timestep, device):
    return tanh_pn({'depth': DEPTH})


def make_values(polarity):
    return (
        gen_rand_tensor(
            polarity, shape=(10000,), width=math.log2(TIMESTEPS)
        ).type(global_config.ntype),
    )


def make_performance_values(polarity):
    low = -1.0 if polarity == 'bipolar' else 0.0
    return (torch.linspace(low, 1.0, 131072, dtype=global_config.ntype),)


def analytic_reference(values, polarity):
    return torch.tanh(values[0] * GAIN)


def known_answer_case(polarity):
    values = torch.tensor([-1.0, 1.0])
    return (values,), torch.tanh(values * GAIN), 0.01


CONFIG = {
    'polarities': ['bipolar'],
    'tolerance_scale': 5.5,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': TIMESTEPS,
}


def test_tanh_pn():
    """Verify tanh_pn against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_tanh_pn()

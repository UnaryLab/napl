import math

import torch
import torch.nn.functional as F

from napl.sim.base import global_config
from napl.sim.module import avgpool2d
from napl.utils._shared_test import streaming_suite


_KERNEL_SIZE = 2
_SHAPE = (4, 3, 8, 8)


def make_operation(polarity, timestep, device):
    return avgpool2d(
        _KERNEL_SIZE,
        config={'polarity': polarity},
    ).to(device)


def make_values(polarity):
    low = 0.0 if polarity == 'unipolar' else -1.0
    return (
        torch.linspace(
            low,
            1.0,
            math.prod(_SHAPE),
            dtype=global_config.ntype,
        ).reshape(_SHAPE),
    )


def make_performance_values(polarity):
    values = make_values(polarity)[0]
    return (values.repeat(256, 1, 1, 1),)


def analytic_reference(values, polarity):
    return F.avg_pool2d(values[0], _KERNEL_SIZE)


def known_answer_case(polarity):
    value = torch.full((1, 1, 4, 4), 0.5, dtype=global_config.ntype)
    return (value,), torch.full((1, 1, 2, 2), 0.5), 0.0


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 4.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
}


def test_avgpool2d():
    """Verify avgpool2d against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_avgpool2d()

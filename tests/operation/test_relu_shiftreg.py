import torch

from napl.sim.operation import relu_shiftreg
from napl.utils._shared_test import streaming_suite


TIMESTEPS = 256


def make_operation(_polarity, _timestep, _device):
    return relu_shiftreg({'depth': 4})


def make_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 1024),)


def make_performance_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 131072),)


def analytic_reference(values, _polarity):
    return torch.relu(values[0])


def known_answer_case(_polarity):
    values = torch.tensor([-1.0, 0.0, 1.0])
    expected = torch.tensor([0.0, 0.0, 1.0])
    return (values,), expected, 0.35


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


def test_relu_shiftreg():
    """Verify relu_shiftreg against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_relu_shiftreg()

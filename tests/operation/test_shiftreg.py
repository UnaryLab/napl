import pytest
import torch

from napl.sim.base import global_config
from napl.sim.operation import shiftreg
from napl.utils._shared_test import devices, streaming_suite


_DEPTH = 2
_TIMESTEPS = 256


def make_operation(polarity, timestep, device):
    return shiftreg({'depth': _DEPTH})


def make_values(polarity):
    low = -1 if polarity == 'bipolar' else 0
    return (
        torch.linspace(low, 1, 512, dtype=global_config.ntype),
    )


def analytic_reference(values, polarity):
    return values[0]


def known_answer_case(polarity):
    return (
        (torch.tensor([1.0]),),
        torch.tensor([1.0]),
        2 * _DEPTH / _TIMESTEPS,
    )


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 1.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': _TIMESTEPS,
}


def test_shiftreg():
    """Verify shiftreg against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        ([1, 1, 1, 0], [0, 1, 1, 1]),
        ([0, 0, 1, 1], [0, 1, 0, 0]),
    ],
)
def test_mutable_input_delay(values, expected):
    """Verify shiftreg delays values correctly when the same input tensor is mutated in place."""
    for device in devices():
        operation = shiftreg({'depth': _DEPTH}).to(device)
        input = torch.zeros(1, dtype=global_config.stype, device=device)
        outputs = []
        for value in values:
            outputs.append(operation(input.fill_(value)).item())
        assert outputs == expected


if __name__ == '__main__':
    test_shiftreg()
    test_mutable_input_delay([1, 1, 1, 0], [0, 1, 1, 1])
    test_mutable_input_delay([0, 0, 1, 1], [0, 1, 0, 0])
    print('Test passed.')

import pytest
import torch

from napl.sim.base import global_config
from napl.sim.operation import dff
from napl.utils._shared_test import devices, streaming_suite


_DEPTH = 3
_TIMESTEPS = 256
# Element count of the per-element delay drive, wide enough to expose a delay line
# that holds its state for only part of the input.
_WIDE = 1024


def make_operation(polarity, timestep, device):
    return dff({'depth': _DEPTH})


def make_values(polarity):
    low = -1 if polarity == 'bipolar' else 0
    return (
        torch.linspace(low, 1, 512, dtype=global_config.ntype),
    )


def make_random_perf_values(polarity):
    low = -1 if polarity == 'bipolar' else 0
    return (
        torch.linspace(low, 1, 131072, dtype=global_config.ntype),
    )


def analytic_reference(values, polarity):
    return values[0]


def known_answer_case(polarity):
    return (
        (torch.tensor([1.0]),),
        torch.tensor([1.0]),
    )


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': _TIMESTEPS,
}


def test_dff():
    """Verify dff against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


def test_wide_delay():
    """Verify every element of a wide input is delayed by depth, which the decoded rate cannot see."""
    torch.manual_seed(0)
    steps = 16
    pattern = torch.randint(0, 2, (steps, _WIDE), dtype=global_config.stype)
    for device in devices():
        operation = dff({'depth': _DEPTH}).to(device)
        for step in range(steps):
            output = operation(pattern[step].to(device)).cpu()
            if step < _DEPTH:
                # The delay line starts empty.
                expected = torch.zeros(_WIDE, dtype=global_config.stype)
            else:
                expected = pattern[step - _DEPTH]
            wrong = int((output != expected).sum())
            assert wrong == 0, f'[{device}] step {step}: {wrong} of {_WIDE} elements not delayed by {_DEPTH}'


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        ([1, 1, 1, 0, 0], [0, 0, 0, 1, 1]),
        ([0, 1, 0, 1, 1], [0, 0, 0, 0, 1]),
    ],
)
def test_mutable_input_delay(values, expected):
    """Verify dff delays values correctly when the same input tensor is mutated in place."""
    for device in devices():
        operation = dff({'depth': _DEPTH}).to(device)
        input = torch.zeros(1, dtype=global_config.stype, device=device)
        outputs = []
        for value in values:
            outputs.append(operation(input.fill_(value)).item())
        assert outputs == expected


if __name__ == '__main__':
    test_dff()
    test_wide_delay()
    test_mutable_input_delay([1, 1, 1, 0, 0], [0, 0, 0, 1, 1])
    test_mutable_input_delay([0, 1, 0, 1, 1], [0, 0, 0, 0, 1])
    print('Test passed.')

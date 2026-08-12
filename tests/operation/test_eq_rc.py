import torch

from napl.sim.base import global_config
from napl.utils._shared_test import streaming_suite
from napl.sim.operation import eq_rc


# Rate-equality band, in decoded value units, shared by the operation and its reference.
_TOLERANCE = 0.1


def make_operation(polarity, _timestep, _device):
    return eq_rc({
        'polarity': polarity,
        'tolerance': _TOLERANCE,
    })


def make_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    left = torch.linspace(lo, hi, 128)
    return left, left.roll(31)


def make_random_perf_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    left = torch.linspace(lo, hi, 131072)
    return left, left.roll(31)


def analytic_reference(values, _polarity):
    return ((values[0] - values[1]).abs() <= _TOLERANCE).type(global_config.ntype)


def known_answer_case(polarity):
    if polarity == 'unipolar':
        values = (torch.tensor([0.5, 1.0]), torch.tensor([0.5, 0.0]))
    else:
        values = (torch.tensor([0.0, 1.0]), torch.tensor([0.0, -1.0]))
    # Equal rates decode toward 1; a full-range gap decodes toward 0.
    return values, torch.tensor([1.0, 0.0])


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'output_polarity': 'unipolar',
    'timesteps': 256,
    # Gate 17 does not apply: eq_rc has a single output that the suite decodes,
    # and an identity wire that copied an input to the output would change the
    # decoded value, so the four suite checks already fail on it.
    'extra_checks': None,
}


def test_eq_rc():
    """Verify eq_rc for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_eq_rc()

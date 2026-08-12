import torch

from napl.sim.operation import clamp
from napl.utils._shared_test import streaming_suite


# Fixed band per polarity, both bounds inside the polarity's legal value range.
BOUNDS = {'bipolar': (-0.5, 0.5), 'unipolar': (0.25, 0.75)}


def make_operation(polarity, _timestep, _device):
    lo, hi = BOUNDS[polarity]
    return clamp({'polarity': polarity, 'lo': lo, 'hi': hi})


def make_values(polarity):
    # Full legal range, so values below lo and above hi are both exercised.
    if polarity == 'bipolar':
        return (torch.linspace(-1.0, 1.0, 128),)
    return (torch.linspace(0.0, 1.0, 128),)


def make_random_perf_values(polarity):
    if polarity == 'bipolar':
        return (torch.linspace(-1.0, 1.0, 131072),)
    return (torch.linspace(0.0, 1.0, 131072),)


def analytic_reference(values, polarity):
    lo, hi = BOUNDS[polarity]
    return torch.clamp(values[0], lo, hi)


def known_answer_case(polarity):
    lo, hi = BOUNDS[polarity]
    if polarity == 'bipolar':
        values = torch.tensor([-0.9, 0.1, 0.9])
    else:
        values = torch.tensor([0.1, 0.5, 0.9])
    # A value below lo clamps up to lo, above hi clamps down to hi, inside passes.
    return (values,), torch.clamp(values, lo, hi)


CONFIG = {
    'polarities': ['bipolar', 'unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    # Gate 17 does not apply: clamp has a single decoded output, and an identity
    # wire decodes to the input, so any value below lo or above hi already makes
    # the known-answer and fidelity checks disagree.
    'extra_checks': None,
    'timesteps': 256,
}


def _config_validation_checks():
    """Assert that an inverted or out-of-range band is rejected on construction."""
    # lo >= hi is rejected.
    for bad in ({'polarity': 'bipolar', 'lo': 0.5, 'hi': -0.5},
                {'polarity': 'bipolar', 'lo': 0.3, 'hi': 0.3}):
        try:
            clamp(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'expected AssertionError for inverted band {bad}')
    # A bound outside the polarity's legal value range is rejected.
    for bad in ({'polarity': 'bipolar', 'lo': -0.5, 'hi': 1.5},
                {'polarity': 'unipolar', 'lo': -0.25, 'hi': 0.75}):
        try:
            clamp(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'expected AssertionError for out-of-range band {bad}')


def test_clamp():
    """Verify clamp against analytic and known-answer streams, and reject invalid bands."""
    streaming_suite(CONFIG)
    _config_validation_checks()


if __name__ == '__main__':
    test_clamp()

import torch

from napl.sim.operation import pow_n
from napl.utils._shared_test import streaming_suite


# Fixed power exercised by the streaming suite. x^3 stays in the legal range for
# both polarities: [0, 1] -> [0, 1] unipolar, [-1, 1] -> [-1, 1] bipolar.
N = 3


def make_operation(polarity, _timestep, _device):
    return pow_n({'polarity': polarity, 'n': N})


def make_values(polarity):
    # Full legal range per polarity; x^N stays in range for either.
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(lo, hi, 128),)


def analytic_reference(values, _polarity):
    return values[0].pow(N)


def known_answer_case(polarity):
    values = torch.tensor([0.0, 0.5, 1.0]) if polarity == 'unipolar' else torch.tensor([-1.0, 0.0, 1.0])
    return (values,), values.pow(N)


def make_random_perf_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(lo, hi, 131072),)


def _validation_checks():
    """Construction rejects non-integer, too-small, and too-large powers."""
    for bad_n in [1, 1.5, pow_n.N_MAX + 1]:
        raised = False
        try:
            pow_n({'polarity': 'bipolar', 'n': bad_n})
        except AssertionError:
            raised = True
        assert raised, f'pow_n accepted illegal n={bad_n}'


# Gate 17 does not apply: pow_n emits a single read output, so any defect changes
# the decoded value the suite already checks. extra_checks instead runs the
# construction-time n-validation asserts after the suite's fidelity checks.
CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    'extra_checks': _validation_checks,
}


def test_pow_n():
    """Verify pow_n raises a stream to n=3 for both polarities and rejects illegal n."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_pow_n()

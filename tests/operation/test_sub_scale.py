import torch

from napl.sim.base import global_config
from napl.sim.operation import sub_scale
from napl.utils._shared_test import devices, streaming_suite


INTWIDTH = 20
FRACWIDTH = 4
SCALE = 2


def make_operation(polarity, _timestep, _device):
    return sub_scale({
        'polarity': polarity,
        'scale': SCALE,
        'intwidth': INTWIDTH,
        'fracwidth': FRACWIDTH,
    })


def make_values(_polarity):
    # Two bipolar inputs a, b. Required range: any a, b with |a - b| <= scale so
    # (a - b) / scale stays in [-1, 1]; with scale=2 the inputs are narrowed to
    # [-0.9, 0.9] to keep the difference strictly inside the bipolar range.
    a = torch.linspace(-0.9, 0.9, 128)
    b = torch.linspace(0.9, -0.9, 128)
    return (a, b)


def make_random_perf_values(_polarity):
    a = -0.9 + 1.8 * torch.rand(16384, 8)
    b = -0.9 + 1.8 * torch.rand(16384, 8)
    return (a, b)


def analytic_reference(values, _polarity):
    return (values[0] - values[1]) / SCALE


def known_answer_case(_polarity):
    # a is the full-scale +1 stream and b the full-scale -1 stream, so (-b) is
    # also +1: the scaled adder of two all-ones streams at scale=2 reproduces the
    # +1 stream exactly, making the answer bit-exact.
    a = torch.ones((4, 4))
    b = -torch.ones((4, 4))
    return (a, b), (a - b) / SCALE


CONFIG = {
    'polarities': ['bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    # Gate 17 does not apply: sub_scale returns a single read output and is not
    # rate-conserving, so a wrong decoded value already catches its failure modes.
    'extra_checks': None,
}


def test_sub_scale():
    """Verify sub_scale computes (a - b) / scale for bipolar streams.

    Bipolar only: the negate-then-add mechanism negates the subtrahend, and a
    unipolar stream carries no negative value to negate.
    """
    streaming_suite(CONFIG)


def test_sub_scale_shape_is_preserved():
    """Verify sub_scale keeps the input shape and ticks the timestep on rank-2 input."""
    torch.manual_seed(0)
    a_cpu = torch.randint(0, 2, (4, 5)).type(global_config.stype)
    b_cpu = torch.randint(0, 2, (4, 5)).type(global_config.stype)
    for device in devices():
        a = a_cpu.to(device)
        b = b_cpu.to(device)
        op = sub_scale({
            'polarity': 'bipolar',
            'scale': SCALE,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        }).to(device)
        output = op(a, b)
        assert output.shape == a.shape
        assert output.dtype == global_config.stype
        assert op.timestep_cur == 1
        op.reset()
        assert op.timestep_cur == 0


def test_sub_scale_rejects_unipolar():
    """Verify sub_scale rejects a unipolar configuration with an AssertionError.

    Bipolar only: negation flips a value's sign, which a unipolar stream cannot
    represent, so the negate-then-add mechanism has no unipolar form.
    """
    try:
        sub_scale({
            'polarity': 'unipolar',
            'scale': SCALE,
            'intwidth': INTWIDTH,
            'fracwidth': FRACWIDTH,
        })
    except AssertionError:
        return
    raise AssertionError('sub_scale must reject unipolar because it negates the subtrahend')


if __name__ == '__main__':
    test_sub_scale()
    test_sub_scale_shape_is_preserved()
    test_sub_scale_rejects_unipolar()
    print('Test passed.')

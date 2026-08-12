import torch

from napl.sim.base import global_config
from napl.sim.operation import or_sat
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256


def make_operation(polarity, _timestep, _device):
    return or_sat({'polarity': polarity})


def make_values(_polarity):
    # Two unipolar inputs a, b. a + b - a*b stays in [0, 1] for any a, b in
    # [0, 1], so the full unipolar range is used with no narrowing.
    a = torch.linspace(0.0, 1.0, 128)
    b = torch.linspace(1.0, 0.0, 128)
    return (a, b)


def make_random_perf_values(_polarity):
    a = torch.rand(16384, 8)
    b = torch.rand(16384, 8)
    return (a, b)


def analytic_reference(values, _polarity):
    return values[0] + values[1] - values[0] * values[1]


def known_answer_case(_polarity):
    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    expected = torch.tensor([0.0, 1.0, 1.0, 1.0])
    return (a, b), expected


CONFIG = {
    'polarities': ['unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': TIMESTEPS,
    # The two operands must be independent for a + b - a*b to hold, so each
    # encoder draws from a distinct Sobol dimension.
    'encoder_dims': lambda _polarity: [1, 2],
    # Gate 17 does not apply: or_sat returns a single read output and is not
    # rate-conserving, so a wrong decoded value already catches its failure modes.
    'extra_checks': None,
}


def test_or_sat():
    """Verify or_sat computes the saturating add a + b - a*b for unipolar streams.

    Unipolar only: a bitwise OR of two bipolar streams carries no saturating-add
    value semantics, so or_sat supports unipolar alone.
    """
    streaming_suite(CONFIG)


def test_or_sat_truth_table():
    """Verify or_sat implements the unipolar OR truth table and ticks the timestep."""
    a = torch.tensor([0, 0, 1, 1], dtype=global_config.stype)
    b = torch.tensor([0, 1, 0, 1], dtype=global_config.stype)
    expected = torch.tensor([0, 1, 1, 1], dtype=global_config.stype)
    for device in devices():
        op = or_sat({'polarity': 'unipolar'}).to(device)
        result = op(a.to(device), b.to(device)).cpu()
        assert torch.equal(result, expected)
        assert result.dtype == global_config.stype
        assert op.timestep_cur == 1
        op.reset()
        assert op.timestep_cur == 0


def test_or_sat_shape_is_preserved():
    """Verify or_sat keeps the input shape and dtype on rank-2 input."""
    torch.manual_seed(0)
    a_cpu = torch.randint(0, 2, (4, 5)).type(global_config.stype)
    b_cpu = torch.randint(0, 2, (4, 5)).type(global_config.stype)
    for device in devices():
        a = a_cpu.to(device)
        b = b_cpu.to(device)
        op = or_sat({'polarity': 'unipolar'}).to(device)
        output = op(a, b)
        assert output.shape == a.shape
        assert output.dtype == global_config.stype
        assert op.timestep_cur == 1
        op.reset()
        assert op.timestep_cur == 0


def test_or_sat_rejects_bipolar():
    """Verify or_sat rejects a bipolar configuration with an AssertionError.

    Unipolar only: a bitwise OR of two bipolar streams has no clean
    saturating-add value semantics.
    """
    try:
        or_sat({'polarity': 'bipolar'})
    except AssertionError:
        return
    raise AssertionError('or_sat must reject bipolar because OR has no bipolar saturating-add semantics')


if __name__ == '__main__':
    test_or_sat()
    test_or_sat_truth_table()
    test_or_sat_shape_is_preserved()
    test_or_sat_rejects_bipolar()
    print('Test passed.')

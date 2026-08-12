import torch

from napl.sim.base import global_config
from napl.sim.operation import and_corr
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256


def make_operation(polarity, timestep, device):
    return and_corr({'polarity': polarity})


def make_values(polarity):
    # Two unipolar operands over the full [0, 1] range. The minimum semantics
    # need no restriction beyond the unipolar range.
    low = 0.0
    return (
        torch.linspace(low, 1.0, 10000, dtype=global_config.ntype),
        torch.linspace(low, 1.0, 10000, dtype=global_config.ntype).roll(3333),
    )


def make_random_perf_values(polarity):
    # >= 1e5 elements for the performance benchmark. Both operands share the
    # legal unipolar range; the suite redraws them at random over that range.
    values = torch.linspace(0.0, 1.0, 131072, dtype=global_config.ntype)
    return values, values.roll(17)


def analytic_reference(values, polarity):
    # AND on maximally-correlated (same Sobol dim) unipolar streams computes the
    # elementwise minimum; on independent streams it would compute the product.
    return torch.minimum(values[0], values[1])


def known_answer_case(polarity):
    # Values on the 1/256 grid so the minimum is exactly representable at N=256.
    values = (
        torch.tensor([0.0, 0.25, 0.50, 1.0]),
        torch.tensor([0.0, 0.75, 0.25, 0.5]),
    )
    expected = torch.tensor([0.0, 0.25, 0.25, 0.5])
    return values, expected


CONFIG = {
    'polarities': ['unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    # Both operands share Sobol dim 1, giving maximal positive correlation
    # (SCC +1). This is what makes AND realize the minimum; distinct dims would
    # decorrelate the streams and the reference would become the product.
    'encoder_dims': lambda polarity: [1, 1],
    # Gate 17 applies: under the correlated same-dim wiring AND(a, a) = a equals an
    # identity wire, so the fidelity run cannot see a kernel-to-wire swap. The
    # identity-wire bite is test_and_corr_truth_table, which drives distinct bits
    # where AND differs from both a wire and OR.
    'timesteps': TIMESTEPS,
}


def test_and_corr():
    """Verify and_corr's execution model, reset/replay, and device coverage; fidelity to min is printed, not asserted."""
    streaming_suite(CONFIG)


def test_and_corr_rejects_bipolar():
    """Verify and_corr rejects a bipolar configuration: AND-as-minimum is a unipolar-only construction."""
    for device in devices():
        try:
            and_corr({'polarity': 'bipolar'}).to(device)
        except AssertionError:
            continue
        raise AssertionError('and_corr accepted a bipolar polarity but must reject it')


def test_and_corr_truth_table():
    """Verify and_corr implements the unipolar AND truth table on distinct bits, the gate-17 identity-wire bite.

    On distinct operand bits AND differs from both an identity wire and OR, so this
    per-timestep bit-exact check fails when the kernel is replaced by a wire, which
    the correlated same-dim fidelity run cannot see.
    """
    a = torch.tensor([0, 0, 1, 1], dtype=global_config.stype)
    b = torch.tensor([0, 1, 0, 1], dtype=global_config.stype)
    expected = torch.tensor([0, 0, 0, 1], dtype=global_config.stype)
    for device in devices():
        op = and_corr({'polarity': 'unipolar'}).to(device)
        result = op(a.to(device), b.to(device)).cpu()
        assert torch.equal(result, expected)
        assert result.dtype == global_config.stype
        assert op.timestep_cur == 1
        op.reset()
        assert op.timestep_cur == 0


if __name__ == '__main__':
    test_and_corr()
    test_and_corr_rejects_bipolar()
    test_and_corr_truth_table()

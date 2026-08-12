import torch

from napl.sim.base import global_config
from napl.sim.operation import counter
from napl.utils._shared_test import devices, streaming_suite


_WIDTH = 3
# Running count saturates at this ceiling.
_MAX_COUNT = 2 ** _WIDTH - 1
_TIMESTEPS = 256


def make_operation(polarity, timestep, device):
    return counter({'width': _WIDTH})


def make_values(polarity):
    # Unipolar rates over the legal [0, 1] span; the counter is non-negative only.
    return (
        torch.linspace(0.0, 1.0, 512, dtype=global_config.ntype),
    )


def make_random_perf_values(polarity):
    return (
        torch.linspace(0.0, 1.0, 131072, dtype=global_config.ntype),
    )


def analytic_reference(values, polarity):
    # For input rate p, the count reaches max_count near t = max_count / p, and the
    # saturation flag is 1 afterward, so the decoded output rate is 1 - max_count / (p * T).
    rate = values[0].clamp_min(1e-9)
    return (1.0 - _MAX_COUNT / (rate * _TIMESTEPS)).clamp(0.0, 1.0)


def known_answer_case(polarity):
    # An all-ones (rate-1) stream increments every timestep, so the count reaches
    # max_count at t = max_count and the flag is 1 for the remaining timesteps.
    ones = (_TIMESTEPS - _MAX_COUNT + 1) / _TIMESTEPS
    return (
        (torch.tensor([1.0]),),
        torch.tensor([ones]),
    )


# Gate 17 assessment: the counter's output IS the stream the suite decodes, and its
# job is to change the rate (decoded reference 1 - max_count/(p*T) differs from the
# input rate p). Replacing it with an identity wire changes that decoded value, so
# the counter is not identity-blind and gate 17's identity-wire check does not apply;
# it has a single output, so the unread-output criterion does not apply either.
CONFIG = {
    'polarities': ['unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': _TIMESTEPS,
}


def test_counter():
    """Verify counter against its known-answer and analytic streams, including reset and timing.

    Only unipolar is supported: a running spike count is non-negative, so there is
    no bipolar (signed-count) form to check.
    """
    streaming_suite(CONFIG)


def test_structural():
    """Assert the counter's execution model, reset-replay, rank-2 shape, and mid-stream clear."""
    for device in devices():
        operation = counter({'width': _WIDTH}).to(device)

        # Execution model: each call advances timestep_cur by exactly one.
        rank2 = torch.ones((4, 6), dtype=global_config.stype, device=device)
        first_trace = []
        for step in range(1, _MAX_COUNT + 4):
            output = operation(rank2)
            assert operation.timestep_cur == step
            assert output.shape == rank2.shape
            assert operation.count.shape == rank2.shape
            first_trace.append(output.detach().cpu().clone())
        # The count saturated at the ceiling and the flag latched to 1.
        assert torch.equal(operation.count, torch.full_like(operation.count, _MAX_COUNT))
        assert torch.equal(first_trace[-1], torch.ones_like(first_trace[-1]))

        # Reset restores the initial state; replaying the inputs reproduces the outputs.
        operation.reset()
        assert operation.timestep_cur == 0
        for step in range(1, _MAX_COUNT + 4):
            replay = operation(rank2).detach().cpu().clone()
            assert torch.equal(first_trace[step - 1], replay)

    print('counter structural checks passed')


def test_clear_count_midstream():
    """Assert clear_count zeroes the running count mid-stream without ending the run."""
    for device in devices():
        operation = counter({'width': _WIDTH}).to(device)
        ones = torch.ones((5,), dtype=global_config.stype, device=device)
        # Drive past the ceiling so the count is saturated and the flag is high.
        for _ in range(_MAX_COUNT + 2):
            operation(ones)
        assert int(operation.count.abs().sum()) == _MAX_COUNT * 5

        step_before = operation.timestep_cur
        operation.clear_count()
        assert int(operation.count.abs().sum()) == 0
        # The run continues: timestep position is untouched and the flag falls until the count rebuilds.
        assert operation.timestep_cur == step_before
        output = operation(ones)
        assert torch.equal(output, torch.zeros_like(output))

    print('counter clear_count mid-stream check passed')


if __name__ == '__main__':
    test_counter()
    test_structural()
    test_clear_count_midstream()
    print('Test passed.')

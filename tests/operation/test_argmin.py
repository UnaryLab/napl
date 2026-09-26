import torch

from napl.sim.base import global_config
from napl.sim.operation import argmin
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256


def make_operation(polarity, timestep, device):
    return argmin({'polarity': polarity, 'width': 16})


def apply_operation(operation, spikes, values):
    return operation(torch.stack(spikes, dim=-1))


def _one_hot_min(values):
    stacked = torch.stack(values, dim=-1)
    winner = torch.argmin(stacked, dim=-1, keepdim=True)
    return torch.zeros_like(stacked).scatter_(-1, winner, 1)


def make_values(polarity):
    low = 0.0 if polarity == 'unipolar' else -1.0
    base = torch.linspace(low, 1.0, 128, dtype=global_config.ntype)
    return base, base.roll(37), base.roll(79)


def make_random_perf_values(polarity):
    low = 0.0 if polarity == 'unipolar' else -1.0
    base = torch.linspace(low, 1.0, 131072, dtype=global_config.ntype)
    return base, base.roll(37001), base.roll(79001)


def analytic_reference(values, polarity):
    return _one_hot_min(values)


def known_answer_case(polarity):
    if polarity == 'unipolar':
        values = (
            torch.tensor([1.0, 0.0, 0.5, 1.0]),
            torch.tensor([0.0, 1.0, 0.5, 1.0]),
            torch.tensor([0.5, 0.5, 0.0, 1.0]),
        )
    else:
        values = (
            torch.tensor([1.0, -1.0, 0.0, 1.0]),
            torch.tensor([-1.0, 1.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.0, -1.0, 1.0]),
        )
    return values, _one_hot_min(values)


def _run_trace(operation, stream, device):
    """Run one explicit spike trace on a selected device."""
    return torch.stack([
        operation.forward_timestep(timestep.to(device)) for timestep in stream
    ])


def _structural_checks():
    """Verify explicit minimum traces, ties, shapes, polarity, and replay."""
    stream = torch.tensor(
        [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 0]],
        ],
        dtype=global_config.stype,
    )
    expected = torch.tensor(
        [
            [[1, 0, 0], [1, 0, 0]],
            [[1, 0, 0], [1, 0, 0]],
            [[1, 0, 0], [1, 0, 0]],
            [[0, 0, 1], [1, 0, 0]],
        ], dtype=global_config.stype,
    )

    for device in devices():
        operation = argmin({'dim': 1, 'width': 4}).to(device)
        first = _run_trace(operation, stream, device)
        # Explicit traces bite min-to-max comparison mutations.
        assert torch.equal(first.cpu(), expected)
        assert torch.equal(
            first.sum(dim=2), torch.ones_like(first.sum(dim=2))
        )
        assert operation.count.shape == stream.shape[1:]
        assert operation.count.dtype == torch.long
        assert operation.hw.pp_delay == 1
        assert first.shape == stream.shape
        assert first.dtype == global_config.stype
        assert first.device.type == device
        assert operation.count.device.type == device

        operation.reset()
        replay = _run_trace(operation, stream, device)
        assert torch.equal(first, replay)
        assert operation.timestep_cur == stream.size(0)

        tied = argmin({'width': 4}).to(device)
        tied_input = torch.tensor(
            [[1, 1, 0], [1, 1, 0]],
            dtype=global_config.stype, device=device,
        )
        tied_trace = torch.stack([
            tied.forward_timestep(tied_input) for _ in range(4)
        ])
        tied_expected = torch.tensor(
            [
                [[1, 0, 0], [1, 0, 0]],
                [[0, 0, 1], [0, 0, 1]],
                [[0, 0, 1], [0, 0, 1]],
                [[0, 0, 1], [0, 0, 1]],
            ], dtype=global_config.stype, device=device,
        )
        assert torch.equal(tied_trace, tied_expected)

        # Saturation is observable only once both lanes pin at count_max: the
        # lower-rate lane then loses the tie to the lower index.
        pinned = argmin({'width': 2}).to(device)
        pinned_stream = torch.tensor(
            [[1, 1], [1, 1], [1, 1], [1, 0], [1, 1]],
            dtype=global_config.stype, device=device,
        )
        for spikes in pinned_stream:
            pinned.forward_timestep(spikes)
        assert torch.equal(pinned.count, torch.tensor([3, 3], device=device))
        assert torch.equal(
            pinned.forward_timestep(pinned_stream[0]),
            torch.tensor([1, 0], dtype=global_config.stype, device=device),
        )

        without_polarity = argmin().to(device)
        with_polarity = argmin({'polarity': 'bipolar'}).to(device)
        assert torch.equal(
            _run_trace(without_polarity, stream, device),
            _run_trace(with_polarity, stream, device),
        )

        rank_one = argmin().to(device)
        rank_one_input = torch.tensor([1, 0, 0], device=device)
        rank_one_output = rank_one(rank_one_input)
        rank_one_next = rank_one(rank_one_input)
        assert rank_one_output.shape == (3,)
        assert torch.equal(
            rank_one_output, torch.tensor([1, 0, 0], device=device)
        )
        assert torch.equal(
            rank_one_next, torch.tensor([0, 1, 0], device=device)
        )
        print(f'[{device}] min traces, ties, shape, polarity, and replay passed')


def _assert_raises(exception, message, callable_):
    """Assert one callable raises the exact expected exception and message."""
    try:
        callable_()
    except exception as error:
        assert str(error) == message
    else:
        raise AssertionError(f'Expected {exception.__name__}: {message}')


def test_argmin_rejects_invalid_config():
    """Verify every construction guard rejects its invalid configuration."""
    defaults = argmin()
    assert defaults.dim == -1
    assert defaults.width == 32
    for config, message in [
        ({'dim': 1.5}, 'Invalid dim: <1.5>; legal values: an integer.'),
        ({'width': 0},
         'Invalid width: <0>; legal values: an integer from 1 to 63.'),
        ({'width': 64},
         'Invalid width: <64>; legal values: an integer from 1 to 63.'),
        ({'width': 2.0},
         'Invalid width: <2.0>; legal values: an integer from 1 to 63.'),
        ({'polarity': 'ternary'},
         "Invalid polarity: <ternary>; legal values: <['unipolar', 'bipolar']>."),
        ({'extra': 0},
         "Unknown key <extra> in the input configuration; accepted keys: <['dim', 'name', 'polarity', 'width']>."),
    ]:
        _assert_raises(AssertionError, message, lambda config=config: argmin(config))


def test_argmin_rejects_invalid_input():
    """Verify input rank, dimension, lane-count, and shape guards."""
    _assert_raises(
        ValueError,
        'argmin input must have rank at least 1.',
        lambda: argmin()(torch.tensor(0)),
    )
    _assert_raises(
        IndexError,
        'argmin dimension 2 is out of range for input with 2 dimensions.',
        lambda: argmin({'dim': 2})(torch.zeros(2, 3)),
    )
    _assert_raises(
        ValueError,
        'argmin requires at least 2 candidate streams.',
        lambda: argmin()(torch.zeros(2, 1)),
    )
    operation = argmin()
    operation(torch.zeros(2, 3))
    _assert_raises(
        ValueError,
        'argmin input shape cannot change before reset.',
        lambda: operation(torch.zeros(3, 3)),
    )


def test_argmin_saturates_bounded_count():
    """Verify integer counters saturate at the configured unsigned maximum."""
    operation = argmin({'width': 2})
    spikes = torch.tensor([[0, 1]], dtype=global_config.stype)
    for _ in range(6):
        operation(spikes)
    assert operation.count.dtype == torch.long
    assert torch.equal(operation.count, torch.tensor([[0, 3]]))
    assert operation.count_max == 3
    assert torch.equal(
        operation(spikes), torch.tensor([[1, 0]], dtype=global_config.stype)
    )
    assert torch.equal(operation.count, torch.tensor([[0, 3]]))


def _suite_config():
    """Return the streaming-suite configuration."""
    return {
        'polarities': ['unipolar', 'bipolar'],
        'make_operation': make_operation,
        'make_values': make_values,
        'make_random_perf_values': make_random_perf_values,
        'analytic_reference': analytic_reference,
        'known_answer_case': known_answer_case,
        'apply_operation': apply_operation,
        'output_polarity': 'unipolar',
        'timesteps': TIMESTEPS,
        'warmup_runs': 0,
        'trials': 1,
    }


def test_argmin():
    """Verify minimum rate selection for both input polarities."""
    streaming_suite(_suite_config())
    _structural_checks()


if __name__ == '__main__':
    test_argmin_rejects_invalid_config()
    test_argmin_rejects_invalid_input()
    test_argmin_saturates_bounded_count()
    test_argmin()

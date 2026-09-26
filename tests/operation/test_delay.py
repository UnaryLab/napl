import pytest
import torch

from napl.sim.base import global_config
from napl.sim.operation import delay
from napl.utils._shared_test import devices, streaming_suite


_DEPTH = 2
_TIMESTEPS = 256
# Element count of the per-element delay drive, wide enough to expose a delay line
# that holds its state for only part of the input.
_WIDE = 1024


def make_operation(polarity, timestep, device):
    return delay({'depth': _DEPTH, 'init': 'alternate'})


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
    # The delay emits the init pattern for the first _DEPTH timesteps and the input
    # stream for the remaining ones, so the decoded value is the input rate diluted
    # over _TIMESTEPS by the warm-up spikes make_operation's 'alternate' init holds.
    warmup_spikes = sum(i % 2 for i in range(_DEPTH))
    rate = (values[0] + 1) / 2 if polarity == 'bipolar' else values[0]
    rate = (rate * (_TIMESTEPS - _DEPTH) + warmup_spikes) / _TIMESTEPS
    return rate * 2 - 1 if polarity == 'bipolar' else rate


def known_answer_case(polarity):
    return (
        (torch.tensor([1.0]),),
        torch.tensor([1.0]),
    )


def _golden_inputs(steps, shape):
    """Build the deterministic 0/1 drive the golden vectors were generated from."""
    count = steps
    for size in shape:
        count *= size
    flat = ((torch.arange(count) * 7) % 5 < 2).to(global_config.stype)
    return flat.reshape(steps, *shape)


# Pinned outputs of both init modes, run on _golden_inputs at the listed depth and
# shape and flattened per timestep. Each row is one timestep.
_GOLDEN = (
    ('zero', 1, (), 'zero init',
     [[0], [1], [0], [0], [1], [0]]),
    ('zero', 3, (2, 2), 'zero init',
     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1],
      [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]]),
    ('alternate', 2, (4,), 'alternate init',
     [[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 1], [0, 1, 0, 0],
      [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 1]]),
    ('alternate', 3, (), 'alternate init',
     [[0], [1], [0], [1], [0], [0], [1]]),
)


def test_golden_vectors():
    """Verify delay reproduces the pinned spike-stream outputs of both init modes bit-exactly."""
    for init, depth, shape, source, rows in _GOLDEN:
        drive = _golden_inputs(len(rows), shape)
        for device in devices():
            operation = delay({'depth': depth, 'init': init}).to(device)
            for step, row in enumerate(rows):
                output = operation(drive[step].to(device)).cpu()
                expected = torch.tensor(row, dtype=global_config.stype).reshape(shape)
                assert torch.equal(output, expected), (
                    f'[{device}] depth {depth}, init {init}, shape {shape}, step {step}: '
                    f'{output.tolist()} does not match the {source} golden {row}'
                )


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': _TIMESTEPS,
    # The suite prints the decoded fidelity error without asserting it, so the
    # shift itself is gated here by the bit-exact golden vectors.
    'extra_checks': test_golden_vectors,
}


def test_delay():
    """Verify delay against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


@pytest.mark.parametrize('init', ['zero', 'alternate'])
def test_wide_delay(init):
    """Verify every element of a wide input is delayed by depth from the configured init pattern."""
    torch.manual_seed(0)
    steps = 16
    pattern = torch.randint(0, 2, (steps, _WIDE), dtype=global_config.stype)
    for device in devices():
        operation = delay({'depth': _DEPTH, 'init': init}).to(device)
        for step in range(steps):
            output = operation(pattern[step].to(device)).cpu()
            if step < _DEPTH:
                fill = step % 2 if init == 'alternate' else 0
                expected = torch.full((_WIDE,), fill, dtype=global_config.stype)
            else:
                expected = pattern[step - _DEPTH]
            wrong = int((output != expected).sum())
            assert wrong == 0, f'[{device}] step {step}: {wrong} of {_WIDE} elements not delayed by {_DEPTH}'


@pytest.mark.parametrize('init', ['zero', 'alternate'])
def test_scalar_reset(init):
    """Verify a scalar delay restores its init pattern and head on reset and replays identically."""
    depth = 3
    drive = _golden_inputs(8, ())
    expected_reg = torch.tensor(
        [i % 2 if init == 'alternate' else 0 for i in range(depth)],
        dtype=global_config.stype,
    )
    for device in devices():
        operation = delay({'depth': depth, 'init': init}).to(device)
        first = [operation(drive[step].to(device)).item() for step in range(len(drive))]
        operation.reset()
        # A scalar input leaves the register shape unchanged, so forward never refills
        # it; the values and the head position are what tell a reset run from a
        # mid-stream one.
        assert operation.reg.shape == (depth,), f'[{device}] register shape after reset'
        assert torch.equal(operation.reg.cpu(), expected_reg), \
            f'[{device}] register after reset: {operation.reg.tolist()} != {expected_reg.tolist()}'
        assert operation.head.shape == (), f'[{device}] head shape after reset'
        assert int(operation.head) == 0, f'[{device}] head after reset: {int(operation.head)}'
        replay = [operation(drive[step].to(device)).item() for step in range(len(drive))]
        assert replay == first, f'[{device}] scalar replay after reset: {replay} != {first}'


def test_device_migration():
    """Verify register state migrates with the module across devices."""
    if len(devices()) < 2:
        pytest.skip('device migration needs a second device')
    operation = delay({'depth': 4})
    operation(torch.ones(2, dtype=global_config.stype))
    operation = operation.to('mps')
    output = operation(torch.ones(2, dtype=global_config.stype, device='mps'))
    assert output.device.type == 'mps', 'delay register state did not migrate to MPS'


def test_float_input_delay():
    """Verify a float spike tensor delays like an stype one and is truncated on the way in."""
    for device in devices():
        operation = delay({'depth': _DEPTH}).to(device)
        outputs = []
        for value in (1.0, 1.0, 1.0, 0.0):
            outputs.append(operation(torch.tensor([value], dtype=global_config.ntype, device=device)).item())
        assert outputs == [0, 0, 1, 1], \
            f'[{device}] float input delayed incorrectly: {outputs}'
        # head starts at 0, so a fractional first input lands in register slot 0.
        fractional = delay({'depth': _DEPTH}).to(device)
        fractional(torch.tensor([1.7], dtype=global_config.ntype, device=device))
        stored = fractional.reg.cpu()
        # The register is allocated at the spike type, so its dtype is set before
        # any input arrives and does not follow the input's.
        assert stored.dtype is torch.int8, \
            f'[{device}] register allocated with dtype {stored.dtype}, not int8'
        assert stored[0].item() == 1, \
            f'[{device}] float input 1.7 stored as {stored[0].item()}, not truncated to 1'


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        ([1, 1, 1, 0], [0, 1, 1, 1]),
        ([0, 0, 1, 1], [0, 1, 0, 0]),
    ],
)
def test_mutable_input_delay(values, expected):
    """Verify delay delays values correctly when the same input tensor is mutated in place."""
    for device in devices():
        operation = delay({'depth': _DEPTH, 'init': 'alternate'}).to(device)
        input = torch.zeros(1, dtype=global_config.stype, device=device)
        outputs = []
        for value in values:
            outputs.append(operation(input.fill_(value)).item())
        assert outputs == expected


def test_invalid_init():
    """Verify an init value outside the two supported patterns is rejected."""
    with pytest.raises(AssertionError):
        delay({'depth': 2, 'init': 'ones'})


if __name__ == '__main__':
    test_delay()
    test_wide_delay('zero')
    test_wide_delay('alternate')
    test_golden_vectors()
    test_scalar_reset('zero')
    test_scalar_reset('alternate')
    # pytest collects this on its own and skips it; a bare script run cannot
    # handle that skip, so the script decides for itself whether to call it.
    if len(devices()) > 1:
        test_device_migration()
    else:
        print(f'only {len(devices())} device available; device-migration check skipped.')
    test_float_input_delay()
    test_mutable_input_delay([1, 1, 1, 0], [0, 1, 1, 1])
    test_mutable_input_delay([0, 0, 1, 1], [0, 1, 0, 0])
    test_invalid_init()
    print('Test passed.')

import pytest
import torch

from napl.sim.base import global_config
from napl.sim.operation import shiftreg
from napl.utils._shared_test import devices, streaming_suite


_DEPTH = 2
_TIMESTEPS = 256


def make_operation(polarity, timestep, device):
    return shiftreg({'depth': _DEPTH})


def make_values(polarity):
    low = -1 if polarity == 'bipolar' else 0
    return (
        torch.linspace(low, 1, 512, dtype=global_config.ntype),
    )


def make_performance_values(polarity):
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
        2 * _DEPTH / _TIMESTEPS,
    )


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 1.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': _TIMESTEPS,
}


def test_shiftreg():
    """Verify shiftreg against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


def test_device_migration():
    """Verify register state migrates with the module across devices."""
    if len(devices()) < 2:
        pytest.skip('device migration needs a second device')
    operation = shiftreg({'depth': 4})
    operation(torch.ones(2, dtype=global_config.stype))
    operation = operation.to('mps')
    output = operation(torch.ones(2, dtype=global_config.stype, device='mps'))
    assert output.device.type == 'mps', 'shiftreg register state did not migrate to MPS'


@pytest.mark.parametrize(
    ('values', 'expected'),
    [
        ([1, 1, 1, 0], [0, 1, 1, 1]),
        ([0, 0, 1, 1], [0, 1, 0, 0]),
    ],
)
def test_mutable_input_delay(values, expected):
    """Verify shiftreg delays values correctly when the same input tensor is mutated in place."""
    for device in devices():
        operation = shiftreg({'depth': _DEPTH}).to(device)
        input = torch.zeros(1, dtype=global_config.stype, device=device)
        outputs = []
        for value in values:
            outputs.append(operation(input.fill_(value)).item())
        assert outputs == expected


def test_masked_hold():
    """Verify a masked-off element re-emits its stored value while an enabled element shifts."""
    # The first phase gives the held lane a distinct value per slot: a lane holding only
    # zeros cannot tell holding from zeroing, and one holding equal values cannot tell a
    # frozen index from an advancing one.
    phases = (
        ([1, 1], [(1, 1), (0, 0)]),
        ([0, 1], [(0, 0), (0, 1), (0, 0)]),
    )
    for device in devices():
        operation = shiftreg({'depth': _DEPTH}).to(device)
        # The enabled lane must track a plain unmasked register fed the same stream.
        reference = shiftreg({'depth': _DEPTH}).to(device)
        held_outputs = []
        for enable, stream in phases:
            mask_enable = torch.tensor(enable, dtype=global_config.stype, device=device)
            for held, shifting in stream:
                output = operation(
                    torch.tensor([held, shifting], dtype=global_config.stype, device=device),
                    mask_enable=mask_enable,
                )
                expected = reference(
                    torch.tensor([shifting], dtype=global_config.stype, device=device)
                )
                assert output[1].item() == expected.item(), \
                    f'[{device}] enabled lane did not delay like an unmasked register'
                if not enable[0]:
                    held_outputs.append(output[0].item())
        assert held_outputs == [1, 1, 1], \
            f'[{device}] held lane emitted {held_outputs} instead of re-emitting its stored one'


def test_masked_shape_mismatch():
    """Reject an enable mask whose shape differs from the input."""
    operation = shiftreg({'depth': _DEPTH})
    with pytest.raises(AssertionError):
        operation(
            torch.zeros(2, dtype=global_config.stype),
            mask_enable=torch.ones(3, dtype=global_config.stype),
        )


if __name__ == '__main__':
    test_shiftreg()
    # pytest collects this on its own and skips it; a bare script run cannot
    # handle that skip, so the script decides for itself whether to call it.
    if len(devices()) > 1:
        test_device_migration()
    test_masked_hold()
    test_masked_shape_mismatch()
    test_mutable_input_delay([1, 1, 1, 0], [0, 1, 1, 1])
    test_mutable_input_delay([0, 0, 1, 1], [0, 1, 0, 0])
    print('Test passed.')

import torch

from napl.sim.base import global_config
from napl.sim.operation import encode
from napl.sim.operation import relu_tc
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256


def make_operation(_polarity, _timestep, _device):
    return relu_tc({'width': 8})


def make_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 1024),)


def make_performance_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 131072),)


def analytic_reference(values, _polarity):
    return torch.relu(values[0])


def known_answer_case(_polarity):
    values = torch.tensor([-1.0, 0.0, 1.0])
    expected = torch.tensor([0.0, 0.0, 1.0])
    return (values,), expected, 0.0


def check_zero_reference():
    """Verify the internal reference is the temporal code of zero, on rank-2 input."""
    codec_config = {
        'polarity': 'bipolar',
        'timestep': TIMESTEPS,
        'generator': 'temporal',
        'dim': 1,
    }
    shape = (2, 3)
    never_rising = torch.full(shape, -1.0, dtype=global_config.ntype)
    zero = torch.zeros(shape, dtype=global_config.ntype)

    for device in devices():
        operation = relu_tc({'width': 8}).to(device)
        stream = encode(codec_config).to(device)
        reference = encode(codec_config).to(device)
        for _ in range(TIMESTEPS):
            # A never-rising input leaves the internal reference as the output.
            output = operation(stream(never_rising.to(device)))
            expected = reference(zero.to(device))
            assert output.shape == shape
            assert torch.equal(output.cpu(), expected.cpu())


CONFIG = {
    'polarities': ['bipolar'],
    'tolerance_scale': 1.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_generators': ['temporal'],
    'extra_checks': check_zero_reference,
    'timesteps': TIMESTEPS,
}


def test_relu_tc():
    """Verify relu_tc against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_relu_tc()

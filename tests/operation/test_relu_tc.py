import torch

from napl.sim.base import global_config
from napl.sim.operation import encode, relu_tc
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256


def make_operation(_polarity, _timestep, _device):
    return relu_tc({'width': 8})


def make_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 1024),)


def make_random_perf_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 131072),)


def analytic_reference(values, _polarity):
    return torch.relu(values[0])


def known_answer_case(_polarity):
    values = torch.tensor([-1.0, 0.0, 1.0])
    expected = torch.tensor([0.0, 0.0, 1.0])
    return (values,), expected


def check_zero_reference():
    """Verify the internal reference is the temporal code of zero, on rank-2 input."""
    codec_config = {
        'polarity': 'bipolar',
        'timestep': TIMESTEPS,
        'generator': 'temporal',
        'dim': 1,
    }
    shape = (2, 3)
    already_fallen = torch.full(shape, -1.0, dtype=global_config.ntype)
    zero = torch.zeros(shape, dtype=global_config.ntype)

    for device in devices():
        operation = relu_tc({'width': 8}).to(device)
        stream = encode(codec_config).to(device)
        reference = encode(codec_config).to(device)
        for _ in range(TIMESTEPS):
            # An input already fallen at cycle 0 leaves the internal reference as the output.
            output = operation(stream(already_fallen.to(device)))
            expected = reference(zero.to(device))
            assert output.shape == shape
            assert torch.equal(output.cpu(), expected.cpu())


CONFIG = {
    'polarities': ['bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_generators': ['temporal'],
    'extra_checks': check_zero_reference,
    'timesteps': TIMESTEPS,
}


def test_relu_tc():
    """Verify relu_tc against analytic and known-answer streams, including reset and timing."""
    # The kernel holds its own zero-reference encoder.
    assert make_operation('bipolar', TIMESTEPS, 'cpu').internal_encode is True
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_relu_tc()

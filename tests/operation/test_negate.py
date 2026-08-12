import torch

from napl.sim.base import global_config
from napl.sim.operation import negate
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256


def make_operation(polarity, _timestep, _device):
    return negate({'polarity': polarity})


def make_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 128),)


def make_random_perf_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 131072),)


def analytic_reference(values, _polarity):
    return -values[0]


def known_answer_case(_polarity):
    values = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    return (values,), -values


CONFIG = {
    'polarities': ['bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': TIMESTEPS,
}


def test_negate():
    """Verify negate negates a bipolar stream against analytic and known-answer references.

    Bipolar only: a unipolar stream carries no negative value, so inverting it
    is a complement rather than a negation.
    """
    streaming_suite(CONFIG)


def test_negate_spike_inversion():
    """Verify negate returns the bit-exact spike complement and preserves rank-3 shape."""
    torch.manual_seed(0)
    spike_cpu = torch.randint(0, 2, (4, 5, 6)).type(global_config.stype)
    for device in devices():
        spike = spike_cpu.to(device)
        op = negate({'polarity': 'bipolar'}).to(device)
        output = op(spike)
        assert output.shape == spike.shape
        assert output.dtype == global_config.stype
        assert torch.equal(output, (1 - spike_cpu).type(global_config.stype).to(device))
        assert op.timestep_cur == 1
        op.reset()
        assert op.timestep_cur == 0
    print('Test passed.')


def test_negate_rejects_unipolar():
    """Verify negate rejects a unipolar configuration with an AssertionError."""
    try:
        negate({'polarity': 'unipolar'})
    except AssertionError:
        return
    raise AssertionError('negate must reject unipolar because it cannot represent a negative value')


if __name__ == '__main__':
    test_negate()
    test_negate_spike_inversion()
    test_negate_rejects_unipolar()

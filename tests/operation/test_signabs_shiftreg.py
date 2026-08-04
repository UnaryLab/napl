import torch

from napl.sim.operation import decode, encode
from napl.sim.operation import signabs_shiftreg
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256


def make_operation(_polarity, _timestep, _device):
    return signabs_shiftreg({'depth': 8})


def make_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 1024),)


def make_performance_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 131072),)


def analytic_reference(values, _polarity):
    return values[0].abs()


def known_answer_case(_polarity):
    values = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    return (values,), values.abs(), 0.2


def check_sign():
    values_cpu = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    expected = -values_cpu.sign()
    config = {
        'polarity': 'bipolar',
        'timestep': TIMESTEPS,
        'generator': 'sobol',
        'dim': 1,
    }
    for device in devices():
        values = values_cpu.to(device)
        enc = encode(config).to(device)
        operation = signabs_shiftreg({'depth': 8}).to(device)
        dec = decode(config).to(device)
        for _ in range(TIMESTEPS):
            sign, _ = operation(enc(values))
            dec(sign)
        torch.testing.assert_close(
            dec.spike_value.cpu(), expected, atol=0.2, rtol=0
        )


CONFIG = {
    'polarities': ['bipolar'],
    'tolerance_scale': 4.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'apply_operation': lambda operation, spikes: operation(*spikes)[1],
    'timesteps': TIMESTEPS,
    'extra_checks': check_sign,
}


def test_signabs_shiftreg():
    """Verify signabs_shiftreg against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_signabs_shiftreg()

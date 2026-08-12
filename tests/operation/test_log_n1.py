import math

import torch

from napl.sim.base import global_config
from napl.utils._shared_test import devices, streaming_suite
from napl.sim.operation import decode, encode, log_n1


# The kernel approximates the five-term Maclaurin truncation of log(1 + x),
# not the exact log, so the analytic reference is that same polynomial.
def _truncated_log1p(x):
    return x - x**2 / 2 + x**3 / 3 - x**4 / 4 + x**5 / 5


def make_operation(polarity, timestep, _device):
    return log_n1({
        'polarity': polarity,
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 1,
    })


def make_values(_polarity):
    # Legal domain is the unipolar range x in [0, 1]; the log argument 1 + x
    # then stays in [1, 2] and is positive throughout, so no value is excluded.
    return (torch.linspace(0.0, 1.0, 128),)


def make_random_perf_values(_polarity):
    return (torch.linspace(0.0, 1.0, 131072),)


def analytic_reference(values, _polarity):
    return _truncated_log1p(values[0])


def known_answer_case(_polarity):
    # log(1 + 0) = 0 exactly; at x = 1 the truncated series gives 0.7833.
    values = torch.tensor([0.0, 1.0])
    return (values,), _truncated_log1p(values)


def _kernel_specific_checks():
    """
    Gate 17: log(1 + x) strictly compresses a high input, so a kernel that
    returned its input unchanged (an identity wire) would fail this band.

    Also verifies timestep advance, reset, and internal_encode across devices.
    """
    codec_config = {'polarity': 'unipolar', 'timestep': 256, 'generator': 'sobol', 'dim': 5}
    input_cpu = torch.ones(4096).type(global_config.ntype)

    for device in devices():
        encoder = encode(codec_config).to(device)
        decoder = decode(codec_config).to(device)
        operation = log_n1({'polarity': 'unipolar', 'timestep': 256,
                            'generator': 'sobol', 'dim': 1}).to(device)
        value = input_cpu.to(device)

        for _ in range(256):
            decoder(operation(encoder(value)))

        decoded = decoder.spike_value.mean().item()
        # Identity would decode to 1.0; log(1 + 1) truncates to 0.7833.
        assert 0.60 < decoded < 0.95, f'[{device}] high-input decode {decoded:.4f} not compressed'

        assert operation.timestep_cur == 256
        assert operation.internal_encode is True
        operation.reset()
        assert operation.timestep_cur == 0
        print(f'[{device}] high-input decode={decoded:.4f} (identity would be 1.0)')

    print('Test passed.')


CONFIG = {
    'polarities': ['unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [5],  # Distinct from the operation's internal dimensions 1..4.
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_log_n1():
    """Verify log_n1 against the truncated-log1p analytic and known-answer streams, with reset, timing, and the gate-17 compression check."""
    # The kernel holds its own coefficient-stream encoder.
    assert make_operation('unipolar', 256, 'cpu').internal_encode is True
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_log_n1()

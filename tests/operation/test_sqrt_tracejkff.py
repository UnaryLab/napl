import torch

from napl.sim.base import global_config
from napl.sim.operation import sqrt_tracejkff
from napl.utils._shared_test import devices, streaming_suite


def make_operation(polarity, timestep, device):
    return sqrt_tracejkff({'polarity': polarity})


def make_values(polarity):
    return (
        torch.linspace(0, 1, 512, dtype=global_config.ntype),
    )


def make_random_perf_values(_polarity):
    return (torch.linspace(0, 1, 131072, dtype=global_config.ntype),)


def analytic_reference(values, polarity):
    return torch.sqrt(values[0])


def known_answer_case(polarity):
    # x = 0 and x = 1 are the fixed points of sqrt; x = 0.25 is where sqrt(x)
    # and x sit furthest apart, 0.250. The tolerance is half that gap, so a
    # kernel that returned its input would fail the case. The largest error
    # measured at these three values across cpu and mps is 0.027344 unipolar
    # and 0.007812 bipolar, so the tolerance clears the working kernel by 4.6x.
    values = torch.tensor([0.0, 0.25, 1.0], dtype=global_config.ntype)
    return (values,), torch.sqrt(values)


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
}


def test_sqrt_tracejkff():
    """Verify sqrt_tracejkff against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


def test_reset_children():
    """Verify sqrt_tracejkff reset clears its JK flip-flop and polarity-conversion children."""
    for device in devices():
        for polarity in CONFIG['polarities']:
            operation = make_operation(polarity, 1, device).to(device)
            operation(torch.ones(4, dtype=global_config.stype, device=device))
            assert operation.jkff.timestep_cur == 1
            if polarity == 'bipolar':
                assert operation.bi2uni.timestep_cur == 1

            operation.reset()
            assert operation.timestep_cur == 0
            assert operation.jkff.timestep_cur == 0
            assert torch.equal(operation.jkff.q, torch.zeros_like(operation.jkff.q))
            if polarity == 'bipolar':
                assert operation.bi2uni.timestep_cur == 0


if __name__ == '__main__':
    test_sqrt_tracejkff()
    test_reset_children()
    print('Test passed.')

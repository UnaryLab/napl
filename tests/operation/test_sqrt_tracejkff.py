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


def analytic_reference(values, polarity):
    return torch.sqrt(values[0])


def known_answer_case(polarity):
    return (
        (torch.tensor([1.0]),),
        torch.tensor([1.0]),
        0.0,
    )


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 5.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
}


def test_sqrt_tracejkff():
    streaming_suite(CONFIG)


def test_reset_children():
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

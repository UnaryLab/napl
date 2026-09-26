import math

import torch

from napl.sim.base import global_config
from napl.sim.operation import encode, mul_ugemm, encode_cond
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256


def make_operation(polarity, timestep, device):
    return encode_cond({
        'polarity': polarity,
        'timestep': timestep,
        'generator': 'sobol',
    })


def apply_operation(operation, spikes, values):
    return operation(spikes[0], values[1])


def make_values(polarity):
    return (
        gen_rand_tensor(
            polarity, shape=(10000,), width=math.log2(TIMESTEPS)
        ).type(global_config.ntype),
        gen_rand_tensor(
            polarity, shape=(10000,), width=math.log2(TIMESTEPS)
        ).type(global_config.ntype),
    )


def make_random_perf_values(polarity):
    low = -1.0 if polarity == 'bipolar' else 0.0
    values = torch.linspace(low, 1.0, 131072, dtype=global_config.ntype)
    return values, values.roll(17)


def analytic_reference(values, polarity):
    return values[0] * values[1]


def known_answer_case(polarity):
    if polarity == 'unipolar':
        values = (
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
        )
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
    else:
        values = (
            torch.tensor([-1.0, -1.0, 1.0, 1.0]),
            torch.tensor([-1.0, 1.0, -1.0, 1.0]),
        )
        expected = torch.tensor([1.0, -1.0, -1.0, 1.0])
    return values, expected


def check_rank2():
    """Verify encode_cond passes a rank-two enabling stream through at value one."""
    config = {
        'polarity': 'bipolar',
        'timestep': 4,
        'generator': 'sobol',
    }
    input_cpu = torch.tensor(
        [[0, 1, 0], [1, 0, 1]],
        dtype=global_config.stype,
    )
    value_cpu = torch.ones((2, 3), dtype=global_config.ntype)

    for device in devices():
        operation = encode_cond(config).to(device)
        result = operation(input_cpu.to(device), value_cpu.to(device))

        assert result.ndim == 2
        assert torch.equal(result.cpu(), input_cpu)
        # The per-element indices broadcast up to the input shape on first use.
        assert operation.seq_idx.shape == input_cpu.shape
        assert operation.seq_idx_inv.shape == input_cpu.shape


def check_matches_mul_ugemm():
    """Verify encode_cond reproduces the mul_ugemm CSG layer bit for bit."""
    timestep = 64
    shape = (3, 5)
    for polarity in ('unipolar', 'bipolar'):
        torch.manual_seed(0)
        stream_value = gen_rand_tensor(
            polarity, shape=shape, width=math.log2(timestep)
        ).type(global_config.ntype)
        operand = gen_rand_tensor(
            polarity, shape=shape, width=math.log2(timestep)
        ).type(global_config.ntype)

        for device in devices():
            source = encode({'polarity': polarity, 'timestep': timestep,
                             'generator': 'sobol', 'dim': 1}).to(device)
            factored = encode_cond({'polarity': polarity, 'timestep': timestep,
                                      'generator': 'sobol'}).to(device)
            existing = mul_ugemm({'polarity': polarity, 'timestep': timestep,
                                  'generator': 'sobol'}).to(device)
            value_device = stream_value.to(device)
            operand_device = operand.to(device)

            for _ in range(timestep):
                spike = source(value_device)
                assert torch.equal(
                    factored(spike, operand_device),
                    existing(spike, operand_device),
                ), f'encode_cond diverged from mul_ugemm on {polarity} / {device}'
            assert torch.equal(factored.seq_idx, existing.seq_idx)
            if polarity == 'bipolar':
                assert torch.equal(factored.seq_idx_inv, existing.seq_idx_inv)


def extra_checks():
    """Run the structural and equivalence checks the decoded value cannot show."""
    check_rank2()
    check_matches_mul_ugemm()


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'apply_operation': apply_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'extra_checks': extra_checks,
    'timesteps': TIMESTEPS,
}


def test_encode_cond():
    """Verify encode_cond against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_encode_cond()

import math

import torch

from napl.sim.base import global_config
from napl.sim.operation import mux_select
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import streaming_suite


TIMESTEPS = 256


def make_operation(polarity, timestep, device):
    return mux_select({'polarity': polarity})


def make_values(polarity):
    # Three decorrelated inputs (c, a, b). The select c is a unipolar rate in
    # [0, 1]; the operands a, b span the legal range for the suite polarity.
    lo = 0.0 if polarity == 'unipolar' else -1.0
    return (
        gen_rand_tensor(
            'unipolar', shape=(10000,), width=math.log2(TIMESTEPS)
        ).type(global_config.ntype),
        (lo + (1.0 - lo) * torch.rand(10000)).type(global_config.ntype),
        (lo + (1.0 - lo) * torch.rand(10000)).type(global_config.ntype),
    )


def make_random_perf_values(polarity):
    lo = 0.0 if polarity == 'unipolar' else -1.0
    select = torch.linspace(0.0, 1.0, 131072, dtype=global_config.ntype)
    a = torch.linspace(lo, 1.0, 131072, dtype=global_config.ntype)
    return select, a, a.roll(29)


def analytic_reference(values, polarity):
    # Value-domain select: p_c * a + (1 - p_c) * b, with p_c the unipolar select value.
    return values[0] * values[1] + (1.0 - values[0]) * values[2]


def known_answer_case(polarity):
    operand = 1.0 if polarity == 'unipolar' else -1.0
    # c toggles selection between a=operand and b=(-operand or 0); output tracks the chosen leg.
    select = torch.tensor([0.0, 1.0, 0.0, 1.0])
    a = torch.tensor([operand, operand, operand, operand])
    b = torch.tensor([-operand, -operand, -operand, -operand]) if polarity == 'bipolar' \
        else torch.tensor([0.0, 0.0, 0.0, 0.0])
    values = (select, a, b)
    expected = select * a + (1.0 - select) * b
    return values, expected


# gate 17 (identity-blind) does not apply: mux_select returns a single output the
# suite reads, and swapping it for a wire changes that output, so a wrong decoded
# value already catches it. No structural identity-wire check is needed.
CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    # Select is unipolar (a rate); operands and output follow the suite polarity.
    'input_polarities': lambda polarity: ['unipolar', polarity, polarity],
    # Distinct Sobol dims so c, a, b are decorrelated.
    'encoder_dims': lambda polarity: [1, 2, 3],
    'timesteps': TIMESTEPS,
}


def test_mux_select():
    """Verify mux_select against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_mux_select()

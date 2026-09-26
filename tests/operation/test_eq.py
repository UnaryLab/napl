import pytest
import torch

from napl.sim.base import global_config
from napl.utils._shared_test import devices, streaming_suite
from napl.sim.operation import eq


# Counter bit width and equality band, in counter steps, shared by the operation
# and its reference.
_WIDTH = 3
_TOLERANCE = 1
_TIMESTEPS = 256


def make_operation(polarity, _timestep, _device):
    return eq({
        'polarity': polarity,
        'width': _WIDTH,
        'tolerance': _TOLERANCE,
    })


def make_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    left = torch.linspace(lo, hi, 128)
    return left, left.roll(31)


def make_random_perf_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    left = torch.linspace(lo, hi, 131072)
    return left, left.roll(31)


def analytic_reference(values, _polarity):
    # The counter drifts by the rate gap per timestep, so it leaves the band
    # after tolerance / gap timesteps and a persistent gap then holds it on a
    # rail. The fraction of a run spent inside the band, which is what the
    # decoded unipolar output reports, is min(1, tolerance / (gap * timesteps)).
    gap = (values[0] - values[1]).abs()
    if _polarity == 'bipolar':
        # A bipolar value v rides a rate (v + 1) / 2, halving the rate gap.
        gap = gap / 2
    band = torch.full_like(gap, float(_TOLERANCE))
    inside = torch.where(gap > 0, band / (gap * _TIMESTEPS), torch.ones_like(gap))
    return inside.clamp(max=1.0).type(global_config.ntype)


def known_answer_case(polarity):
    if polarity == 'unipolar':
        values = (torch.tensor([0.5, 1.0]), torch.tensor([0.5, 0.0]))
    else:
        values = (torch.tensor([0.0, 1.0]), torch.tensor([0.0, -1.0]))
    # Equal streams keep the counter at half scale and decode toward 1; a
    # full-range gap rails the counter and decodes toward 0.
    return values, torch.tensor([1.0, 0.0])


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'output_polarity': 'unipolar',
    'timesteps': _TIMESTEPS,
    # Gate 17 does not apply: eq returns a single output, which the suite reads,
    # so there is no unread output to pin, and an identity wire would return the
    # input spikes the suite decodes rather than the band indicator.
    'extra_checks': None,
}


# One hand-traced spike-pair sequence at width 3 (half = 4, rails 0 and 7) and
# tolerance 1 (band = counter in {3, 4, 5}), covering a step up, a step down, a
# hold on 00, a hold on 11, both rails saturating, and re-entry into the band
# from above and from below. Lane 1 holds on 00 throughout, so it stays at half.
_TRACE_INPUT_0 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
_TRACE_INPUT_1 = [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
# Counter after each step: 5 5 6 7 7 7 6 5 4 3 2 1 0 0 0 1 2 3.
_TRACE_OUTPUT = [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]


def test_eq():
    """Verify eq for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


def test_eq_counter_trace():
    """Verify eq reproduces a hand-traced counter sequence bit-exactly on every device."""
    for device in devices():
        operation = eq({'width': _WIDTH, 'tolerance': _TOLERANCE}).to(device)
        operation.reset()
        emitted = []
        for spike_0, spike_1 in zip(_TRACE_INPUT_0, _TRACE_INPUT_1):
            input_0 = torch.tensor([[spike_0], [0]], dtype=operation.stype, device=device)
            input_1 = torch.tensor([[spike_1], [0]], dtype=operation.stype, device=device)
            emitted.append(operation(input_0, input_1))
        assert operation.cnt.shape == (2, 1), f'counter shape {operation.cnt.shape}, expected (2, 1)'
        output = torch.cat(emitted, dim=1)
        assert output.shape == (2, len(_TRACE_OUTPUT)), \
            f'output shape {output.shape}, expected (2, {len(_TRACE_OUTPUT)})'
        expected = torch.tensor(
            [_TRACE_OUTPUT, [1] * len(_TRACE_OUTPUT)], dtype=operation.stype, device=device)
        assert torch.equal(output, expected), \
            f'{device}: got {output.tolist()}, expected {expected.tolist()}'
        print(f'{device}: hand-traced counter sequence reproduced bit-exactly.')
    print('Test passed.')


def test_eq_rejects_invalid_config():
    """Verify the width and tolerance guards reject their invalid configurations."""
    # cnt_half is 4 at width 3, so 5 is the first tolerance past the band's upper bound.
    for config, field in [
        ({'width': _WIDTH, 'tolerance': -1}, 'tolerance'),
        ({'width': _WIDTH, 'tolerance': 2**(_WIDTH - 1) + 1}, 'tolerance'),
        ({'width': _WIDTH, 'tolerance': 0.5}, 'tolerance'),
        ({'width': _WIDTH, 'tolerance': True}, 'tolerance'),
        ({'width': 0, 'tolerance': _TOLERANCE}, 'width'),
        ({'width': 3.0, 'tolerance': _TOLERANCE}, 'width'),
    ]:
        with pytest.raises(AssertionError, match=f'Invalid {field}'):
            eq(config)


if __name__ == '__main__':
    test_eq()
    test_eq_counter_trace()
    test_eq_rejects_invalid_config()

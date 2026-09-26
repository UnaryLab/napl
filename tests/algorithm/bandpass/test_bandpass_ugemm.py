import torch

from napl.sim.algorithm.bandpass import bandpass_ugemm
from napl.sim.base import global_config
from napl.sim.operation import add_scale, mul_ugemm
from napl.utils._shared_test import devices, streaming_suite


_TAPS = torch.tensor([0.25, 0.0, -0.25, 0.0, 0.25], dtype=global_config.ntype)
_SCALE = 1.0
_TIMESTEPS = 64


def _config(timestep):
    return {
        'polarity': 'bipolar',
        'timestep': timestep,
        'generator': 'sobol',
        'scale': _SCALE,
        'intwidth': 10,
        'fracwidth': 8,
    }


def make_operation(polarity, timestep, device):
    return bandpass_ugemm(_TAPS, _config(timestep))


def make_values(polarity):
    return (torch.linspace(-1, 1, 512, dtype=global_config.ntype),)


def make_random_perf_values(polarity):
    return (torch.linspace(-1, 1, 131072, dtype=global_config.ntype),)


def analytic_reference(values, polarity):
    return values[0] * (_TAPS.sum() / _SCALE)


def known_answer_case(polarity):
    values = torch.ones(8, dtype=global_config.ntype)
    return ((values,), values * (_TAPS.sum() / _SCALE))


CONFIG = {
    # Signed bandpass taps require bipolar input and output streams.
    'polarities': ['bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': _TIMESTEPS,
    'warmup_runs': 1,
    'trials': 3,
}


def test_bandpass_ugemm():
    """Verify the bipolar-only FIR because signed taps require signed input streams."""
    streaming_suite(CONFIG)


def test_delay_line_timing():
    """Verify the delayed tap against the declared multiplier-and-adder composition."""
    taps = torch.tensor([0.0, 1.0], dtype=global_config.ntype)
    pattern = [1, 0, 1, 1, 0, 0, 1, 0]
    for device in devices():
        operation = bandpass_ugemm(taps, _config(16)).to(device)
        multipliers = [mul_ugemm({
            'polarity': 'bipolar',
            'timestep': 16,
            'generator': 'sobol',
        }).to(device) for _ in range(2)]
        accumulator = add_scale({
            'polarity': 'bipolar',
            'scale': _SCALE,
            'intwidth': 10,
            'fracwidth': 8,
        }).to(device)
        input = torch.zeros(1, dtype=global_config.stype, device=device)
        previous = torch.zeros_like(input)
        output = []
        expected = []
        for value in pattern:
            input.fill_(value)
            output.append(operation(input).item())
            products = torch.stack([
                multipliers[0](input, taps[0].to(device)),
                multipliers[1](previous, taps[1].to(device)),
            ])
            expected.append(accumulator(products, dim=0).item())
            previous.copy_(input)
        assert output == expected, f'[{device}] delayed trace {output} != {expected}'


def test_bandpass_ugemm_rejects_invalid_config():
    """Verify all bandpass-specific constructor guards reject invalid configuration."""
    try:
        bandpass_ugemm(_TAPS, dict(_config(16), polarity='unipolar'))
    except AssertionError as error:
        assert str(error) == "Invalid polarity: <unipolar>; legal values: <['bipolar']>.", error
    else:
        raise AssertionError('bandpass_ugemm accepted unipolar streams')

    for taps, description in (([0.25], 'list'), (torch.ones(1, 2), '(1, 2)')):
        try:
            bandpass_ugemm(taps, _config(16))
        except AssertionError as error:
            assert str(error) == (
                f'Invalid taps: <{description}>; legal values: a non-empty 1-D tensor.'
            ), error
        else:
            raise AssertionError(f'bandpass_ugemm accepted invalid taps {description}')

    try:
        bandpass_ugemm(torch.tensor([1.01]), _config(16))
    except AssertionError as error:
        assert str(error) == (
            'Invalid taps: all coefficients must be finite and in [-1, 1].'
        ), error
    else:
        raise AssertionError('bandpass_ugemm accepted a tap outside [-1, 1]')

    taps = torch.tensor([0.5, 0.6], dtype=global_config.ntype)
    tap_norm = taps.abs().sum().item()
    try:
        bandpass_ugemm(taps, dict(_config(16), scale=1.0))
    except AssertionError as error:
        assert str(error) == (
            f'Invalid scale: <1.0>; legal values: a positive finite number at least '
            f'<{tap_norm}>.'
        ), error
    else:
        raise AssertionError('bandpass_ugemm accepted a scale below the tap L1 norm')

    taps = torch.tensor([0.5, 0.501], dtype=global_config.ntype)
    tap_norm = taps.abs().sum().item()
    try:
        bandpass_ugemm(taps, dict(_config(16), scale=1.001))
    except AssertionError as error:
        assert str(error) == (
            f'Quantized scale <1.0> is below the tap L1 norm <{tap_norm}>; '
            'increase scale or fracwidth.'
        ), error
    else:
        raise AssertionError('bandpass_ugemm accepted a quantized scale below the tap L1 norm')


def test_rank_and_state_shapes():
    """Verify rank-2 input shape is preserved by every persistent streaming state."""
    for device in devices():
        operation = bandpass_ugemm(_TAPS, _config(16)).to(device)
        input = torch.ones(2, 3, dtype=global_config.stype, device=device)
        output = operation(input)
        assert output.shape == input.shape
        assert operation.acc.accumulator.shape == input.shape
        for index in range(operation.tap_count):
            assert getattr(operation, f'mul_{index}').seq_idx.shape == input.shape
            assert getattr(operation, f'mul_{index}').seq_idx_inv.shape == input.shape


if __name__ == '__main__':
    test_bandpass_ugemm()
    test_delay_line_timing()
    test_bandpass_ugemm_rejects_invalid_config()
    test_rank_and_state_shapes()
    print('Test passed.')

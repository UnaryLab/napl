"""Test subabs streaming; gradients are exempt because it has no parameters."""

import torch

from napl.sim.base import global_config
from napl.sim.metric import correlation
from napl.sim.operation import decode, encode, subabs
from napl.utils._shared_test import devices, streaming_suite


def make_operation(polarity, _timestep, _device):
    return subabs({'polarity': polarity})


def make_values(_polarity):
    first = torch.linspace(0.0, 1.0, 128)
    second = torch.linspace(1.0, 0.0, 128)
    return first, second


def make_performance_values(_polarity):
    first = torch.linspace(0.0, 1.0, 131072)
    second = torch.linspace(1.0, 0.0, 131072)
    return first, second


def analytic_reference(values, _polarity):
    return (values[0] - values[1]).abs()


def known_answer_case(_polarity):
    values = (torch.tensor([0.0, 1.0, 0.5, 0.25]),
              torch.tensor([1.0, 0.0, 0.5, 0.75]))
    expected = torch.tensor([1.0, 1.0, 0.0, 0.5])
    return values, expected, 1.0 / 256


CONFIG = {
    # subabs is unipolar only: an XOR gate has no bipolar rate-domain meaning.
    'polarities': ['unipolar'],
    'tolerance_scale': 2.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    # A shared Sobol dimension makes the two operand streams SCC +1, the
    # correlation under which the XOR gate realizes the absolute difference.
    'encoder_dims': [1, 1],
    'timesteps': 256,
}


def test_subabs_config():
    """Reject a bipolar polarity, an absent polarity, and an unknown config key."""
    for config, message in [
            ({'polarity': 'bipolar'},
             'Invalid polarity: <bipolar>; subabs supports unipolar only.'),
            ({},
             'Missing key <polarity> in the input configuration.'),
            ({'polarity': 'unipolar', 'depth': 1},
             "Unknown key <depth> in the input configuration; accepted keys: <['name', 'polarity']>.")]:
        try:
            subabs(config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'subabs accepted the invalid config <{config}>.')


def test_subabs_truth_table():
    """Match the XOR truth table on a rank-2 spike tensor at every device."""
    input_0_cpu = torch.tensor([[0, 0], [1, 1]], dtype=global_config.stype)
    input_1_cpu = torch.tensor([[0, 1], [0, 1]], dtype=global_config.stype)
    expected_cpu = torch.tensor([[0, 1], [1, 0]], dtype=global_config.stype)
    for device in devices():
        operation = subabs({'polarity': 'unipolar'}).to(device)
        output = operation(input_0_cpu.to(device), input_1_cpu.to(device))
        assert output.shape == expected_cpu.shape, output.shape
        assert torch.equal(output.cpu(), expected_cpu), output
        assert operation.timestep_cur == 1
        operation.reset()
        assert operation.timestep_cur == 0


def test_subabs_correlation_sensitivity():
    """Realize the absolute difference for SCC +1 operands and overshoot it otherwise."""
    timesteps = 256
    value_0 = torch.tensor([[0.25, 0.75, 0.5]]).type(global_config.ntype)
    value_1 = torch.tensor([[0.75, 0.25, 0.25]]).type(global_config.ntype)
    reference = (value_0 - value_1).abs()
    for device in devices():
        results = {}
        for label, dims in (('correlated', (1, 1)), ('decorrelated', (1, 3))):
            codec = {'polarity': 'unipolar', 'timestep': timesteps, 'generator': 'sobol'}
            encoder_0 = encode({**codec, 'dim': dims[0]}).to(device)
            encoder_1 = encode({**codec, 'dim': dims[1]}).to(device)
            decoder = decode({**codec, 'dim': 1}).to(device)
            metric = correlation().to(device)
            operation = subabs({'polarity': 'unipolar'}).to(device)
            for _ in range(timesteps):
                spike_0 = encoder_0(value_0.to(device))
                spike_1 = encoder_1(value_1.to(device))
                metric(spike_0, spike_1)
                decoder(operation(spike_0, spike_1))
            results[label] = (decoder.spike_value.cpu(), metric.correlation.mean().item())

        correlated_value, correlated_scc = results['correlated']
        decorrelated_value, decorrelated_scc = results['decorrelated']
        assert correlated_scc > 0.99, correlated_scc
        assert abs(decorrelated_scc) < 0.1, decorrelated_scc
        # Positively correlated operands give |p0 - p1| within one quantization step.
        assert (correlated_value - reference).abs().max().item() <= 1.0 / timesteps, correlated_value
        # Decorrelated operands give p0 + p1 - 2 p0 p1, which is strictly larger here.
        assert (decorrelated_value - reference).min().item() > 0.1, decorrelated_value
        print(f'[{device}] correlated_scc={correlated_scc:.4f}, '
              f'decorrelated_scc={decorrelated_scc:.4f}, '
              f'correlated={correlated_value.tolist()}, '
              f'decorrelated={decorrelated_value.tolist()}')


def test_subabs():
    """Verify subabs against |p0 - p1| for positively correlated unipolar streams."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_subabs_config()
    test_subabs_truth_table()
    test_subabs_correlation_sensitivity()
    test_subabs()
    print('Test passed.')

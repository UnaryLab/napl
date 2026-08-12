"""Test desync streaming; gradients are exempt because it has no parameters."""

import torch

from napl.sim.base import global_config
from napl.sim.metric import correlation
from napl.sim.operation import decode, desync, encode
from napl.utils._shared_test import devices, streaming_suite


# Fig. 3b of the DATE 2018 paper, walked from the initial state S0 with save
# depth 1. Each row is (input_0, input_1, output_0, output_1). Steps 3 and 6
# show the alternation: the released one lands on input_0's output first and on
# input_1's output next.
FSM_D1_SEQUENCE = [
    (1, 1, 0, 1),  # S0 -> S1, save the paired input_0 one
    (1, 1, 1, 1),  # S1 -> S1, no room left, pass through
    (0, 0, 1, 0),  # S1 -> S2, emit the saved input_0 one
    (1, 0, 1, 0),  # S2 -> S2, already unpaired, pass through
    (1, 1, 1, 0),  # S2 -> S3, save the paired input_1 one
    (0, 0, 0, 1),  # S3 -> S0, emit the saved input_1 one
    (0, 0, 0, 0),  # S0 -> S0, nothing saved
    (1, 1, 0, 1),  # S0 -> S1, save the paired input_0 one
    (0, 1, 0, 1),  # S1 -> S1, already unpaired, pass through
    (0, 0, 1, 0),  # S1 -> S2, emit the saved input_0 one
    (1, 1, 1, 0),  # S2 -> S3, save the paired input_1 one
    (1, 1, 1, 1),  # S3 -> S3, no room left, pass through
    (0, 0, 0, 1),  # S3 -> S0, emit the saved input_1 one
]


def make_operation(polarity, _timestep, _device):
    return desync({'polarity': polarity, 'depth': 1})


def make_values(polarity, count=128):
    if polarity == 'bipolar':
        return torch.linspace(-1.0, 1.0, count), torch.linspace(1.0, -1.0, count)
    return torch.linspace(0.0, 1.0, count), torch.linspace(1.0, 0.0, count)


def make_random_perf_values(polarity):
    return make_values(polarity, count=131072)


def analytic_reference(values, _polarity):
    return values[0]


def known_answer_case(polarity):
    if polarity == 'bipolar':
        values = (torch.tensor([-1.0, 1.0, 0.0]), torch.tensor([1.0, -1.0, -0.5]))
        return values, values[0]
    values = (torch.tensor([0.0, 1.0, 0.5]), torch.tensor([1.0, 0.0, 0.25]))
    return values, values[0]


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'apply_operation': lambda operation, spikes: operation(*spikes)[0],
    'timesteps': 256,
}


def _run_pair(operation, value_0, value_1, timesteps, device):
    codec = {'polarity': 'unipolar', 'timestep': timesteps, 'generator': 'sobol'}
    encoder_0 = encode({**codec, 'dim': 1}).to(device)
    encoder_1 = encode({**codec, 'dim': 3}).to(device)
    decoder_0 = decode({**codec, 'dim': 1}).to(device)
    decoder_1 = decode({**codec, 'dim': 1}).to(device)
    metric_in = correlation().to(device)
    metric_out = correlation().to(device)
    for _ in range(timesteps):
        spike_0 = encoder_0(value_0)
        spike_1 = encoder_1(value_1)
        metric_in(spike_0, spike_1)
        if operation is None:
            output_0, output_1 = spike_0, spike_1
        else:
            output_0, output_1 = operation(spike_0, spike_1)
        metric_out(output_0, output_1)
        decoder_0(output_0)
        decoder_1(output_1)
    return (metric_in.correlation.mean().item(),
            metric_out.correlation.mean().item(),
            decoder_0.spike_value, decoder_1.spike_value)


def test_desync_config():
    """Reject an illegal polarity, missing keys, unknown keys, and an invalid depth."""
    depth_message = 'Invalid depth: <{}>; legal values: an integer of at least 1.'
    for config, message in [
            ({'polarity': 'ternary', 'depth': 1},
             "Invalid polarity: <ternary>; legal values: <['unipolar', 'bipolar']>."),
            ({'depth': 1},
             'Missing key <polarity> in the input configuration.'),
            ({'polarity': 'unipolar'},
             'Missing key <depth> in the input configuration.'),
            ({'polarity': 'unipolar', 'depth': 1, 'width': 2},
             "Unknown key <width> in the input configuration; accepted keys: <['depth', 'name', 'polarity']>."),
            ({'polarity': 'unipolar', 'depth': 0}, depth_message.format(0)),
            ({'polarity': 'unipolar', 'depth': -2}, depth_message.format(-2)),
            ({'polarity': 'unipolar', 'depth': 1.5}, depth_message.format(1.5)),
            ({'polarity': 'unipolar', 'depth': True}, depth_message.format(True)),
            ({'polarity': 'unipolar', 'depth': '1'}, depth_message.format('1'))]:
        try:
            desync(config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'desync accepted the invalid config <{config}>.')


def test_desync_known_sequence():
    """Reproduce the Fig. 3b depth-1 state machine step by step on a rank-2 input under both polarities."""
    for device in devices():
        # Polarity only reinterprets the rate, so both polarities emit these bits.
        for polarity in ('unipolar', 'bipolar'):
            operation = desync({'polarity': polarity, 'depth': 1}).to(device)
            assert operation.polarity_io['output_0'] == polarity
            for step, (bit_0, bit_1, expected_0, expected_1) in enumerate(FSM_D1_SEQUENCE, start=1):
                # Both rows carry the same bits, so the per-element machines stay in step.
                input_0 = torch.full((2, 3), bit_0, dtype=global_config.stype).to(device)
                input_1 = torch.full((2, 3), bit_1, dtype=global_config.stype).to(device)
                output_0, output_1 = operation(input_0, input_1)
                assert output_0.shape == (2, 3), output_0.shape
                assert output_1.shape == (2, 3), output_1.shape
                assert operation.cnt.shape == (2, 3), operation.cnt.shape
                assert torch.equal(output_0.cpu(), torch.full((2, 3), expected_0, dtype=global_config.stype)), \
                    f'{polarity} step {step}: input=({bit_0},{bit_1}) output_0={output_0.cpu().tolist()}'
                assert torch.equal(output_1.cpu(), torch.full((2, 3), expected_1, dtype=global_config.stype)), \
                    f'{polarity} step {step}: input=({bit_0},{bit_1}) output_1={output_1.cpu().tolist()}'
                assert operation.timestep_cur == step
            operation.reset()
            assert operation.timestep_cur == 0
            assert operation.cnt.numel() == 1 and operation.cnt.item() == 0
            assert operation.side.numel() == 1 and operation.side.item() == 1


def test_desync_correlation():
    """Drive the output SCC toward -1 while preserving both input values."""
    timesteps = 256
    torch.manual_seed(0)
    value_cpu_0 = torch.rand(4, 64).mul(0.9).add(0.05).type(global_config.ntype)
    value_cpu_1 = torch.rand(4, 64).mul(0.9).add(0.05).type(global_config.ntype)
    for device in devices():
        value_0 = value_cpu_0.to(device)
        value_1 = value_cpu_1.to(device)
        input_scc, base_scc, _, _ = _run_pair(None, value_0, value_1, timesteps, device)
        assert abs(base_scc) < 0.1, base_scc

        previous = None
        for depth in [1, 2, 4]:
            operation = desync({'polarity': 'unipolar', 'depth': depth}).to(device)
            _, output_scc, out_0, out_1 = _run_pair(operation, value_0, value_1, timesteps, device)
            error_0 = (out_0 - value_0).abs().max().item()
            error_1 = (out_1 - value_1).abs().max().item()
            print(f'[{device}][depth={depth}] input_scc={input_scc:.4f}, '
                  f'output_scc={output_scc:.4f}, '
                  f'value_error=({error_0:.6f}, {error_1:.6f})')
            assert output_scc < -0.5, output_scc
            assert output_scc < input_scc - 0.5, (input_scc, output_scc)
            # Only ones still saved at the end of the run are lost, at most depth each.
            assert error_0 <= depth / timesteps + 1e-6, error_0
            assert error_1 <= depth / timesteps + 1e-6, error_1
            if previous is not None:
                assert output_scc <= previous + 1e-6, (previous, output_scc)
            previous = output_scc


def test_desync():
    """Verify desync preserves the first stream value across the full unipolar and bipolar ranges."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_desync_config()
    test_desync_known_sequence()
    test_desync_correlation()
    test_desync()
    print('Test passed.')

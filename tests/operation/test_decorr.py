"""Test decorr streaming; gradients are exempt because it has no parameters."""

import torch

from napl.sim.base import global_config
from napl.sim.metric import correlation
from napl.sim.operation import decode, decorr, encode
from napl.utils._shared_test import devices, streaming_suite


# Fig. 4b of the DATE 2018 paper with buffer depth 4: three storage cells reset to
# the alternating pattern (0, 1, 0) plus a pass-through position. The temporal
# generator makes the selected position ramp 0, 1, 2, 3 and repeat, so position 3
# passes the current input through and positions 0, 1, 2 emit the bit stored three
# timesteps earlier. Each row is (input_0, input_1, output_0, output_1).
SHUFFLE_D4_SEQUENCE = [
    (1, 0, 0, 0),  # position 0 emits its reset 0 and stores the inputs
    (1, 1, 1, 1),  # position 1 emits its reset 1 and stores the inputs
    (0, 1, 0, 0),  # position 2 emits its reset 0 and stores the inputs
    (1, 0, 1, 0),  # position 3 passes both inputs through
    (0, 1, 1, 0),  # position 0 emits the bits stored at step 1
    (1, 0, 1, 1),  # position 1 emits the bits stored at step 2
    (1, 0, 0, 1),  # position 2 emits the bits stored at step 3
    (0, 1, 0, 1),  # position 3 passes both inputs through
]


def make_operation(polarity, _timestep, _device):
    return decorr({'polarity': polarity, 'depth': 4, 'timestep': 256,
                           'generator': 'sys', 'seed': 7})


def make_values(_polarity):
    first = torch.linspace(0.0, 1.0, 128)
    second = torch.linspace(1.0, 0.0, 128)
    return first, second


def make_performance_values(_polarity):
    first = torch.linspace(0.0, 1.0, 131072)
    second = torch.linspace(1.0, 0.0, 131072)
    return first, second


def analytic_reference(values, _polarity):
    return values[0]


def known_answer_case(_polarity):
    values = (torch.tensor([0.0, 1.0, 0.5]), torch.tensor([1.0, 0.0, 0.25]))
    # Three stored bits are lost at the end of the run and three reset bits take
    # their place, on top of the encoder's own resolution.
    return values, values[0], 5.0 / 256


CONFIG = {
    # decorr is unipolar only: the buffer emits stored 0/1 bits whose
    # rate meaning is unipolar.
    'polarities': ['unipolar'],
    'tolerance_scale': 2.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'apply_operation': lambda operation, spikes: operation(*spikes)[0],
    'timesteps': 256,
}


def _run_pair(operation, value_0, value_1, timesteps, device):
    # One shared Sobol dimension makes the two input streams maximally correlated.
    codec = {'polarity': 'unipolar', 'timestep': timesteps, 'generator': 'sobol', 'dim': 1}
    encoder_0 = encode(codec).to(device)
    encoder_1 = encode(codec).to(device)
    decoder_0 = decode(codec).to(device)
    decoder_1 = decode(codec).to(device)
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


def test_decorr_config():
    """Reject a bipolar polarity, missing keys, unknown keys, and an invalid depth."""
    depth_message = 'Invalid depth: <{}>; legal values: an integer of at least 1.'
    for config, message in [
            ({'polarity': 'bipolar', 'depth': 4, 'timestep': 256, 'generator': 'sys'},
             'Invalid polarity: <bipolar>; decorr supports unipolar only.'),
            ({'depth': 4, 'timestep': 256, 'generator': 'sys'},
             'Missing key <polarity> in the input configuration.'),
            ({'polarity': 'unipolar', 'timestep': 256, 'generator': 'sys'},
             'Missing key <depth> in the input configuration.'),
            ({'polarity': 'unipolar', 'depth': 4, 'generator': 'sys'},
             'Missing key <timestep> in the input configuration.'),
            ({'polarity': 'unipolar', 'depth': 4, 'timestep': 256},
             'Missing key <generator> in the input configuration.'),
            ({'polarity': 'unipolar', 'depth': 4, 'timestep': 256, 'generator': 'sys',
              'width': 2},
             "Unknown key <width> in the input configuration; accepted keys: "
             "<['depth', 'dim', 'generator', 'name', 'polarity', 'seed', 'taps', 'timestep']>."),
            ({'polarity': 'unipolar', 'depth': 0, 'timestep': 256, 'generator': 'sys'},
             depth_message.format(0)),
            ({'polarity': 'unipolar', 'depth': -2, 'timestep': 256, 'generator': 'sys'},
             depth_message.format(-2)),
            ({'polarity': 'unipolar', 'depth': 1.5, 'timestep': 256, 'generator': 'sys'},
             depth_message.format(1.5)),
            ({'polarity': 'unipolar', 'depth': True, 'timestep': 256, 'generator': 'sys'},
             depth_message.format(True)),
            ({'polarity': 'unipolar', 'depth': '1', 'timestep': 256, 'generator': 'sys'},
             depth_message.format('1'))]:
        try:
            decorr(config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'decorr accepted the invalid config <{config}>.')


def test_decorr_known_sequence():
    """Reproduce the Fig. 4b depth-4 shuffle buffer step by step on a rank-2 input."""
    for device in devices():
        operation = decorr({'polarity': 'unipolar', 'depth': 4, 'timestep': 4,
                                    'generator': 'tc'}).to(device)
        assert operation.rand_seq_idx == [[0, 1, 2, 3], [0, 1, 2, 3]], operation.rand_seq_idx
        collected = []
        for step, (bit_0, bit_1, expected_0, expected_1) in enumerate(SHUFFLE_D4_SEQUENCE, start=1):
            # Both rows carry the same bits, so the per-element buffers stay in step.
            input_0 = torch.full((2, 3), bit_0, dtype=global_config.stype).to(device)
            input_1 = torch.full((2, 3), bit_1, dtype=global_config.stype).to(device)
            output_0, output_1 = operation(input_0, input_1)
            assert output_0.shape == (2, 3), output_0.shape
            assert output_1.shape == (2, 3), output_1.shape
            assert operation.reg.shape == (2, 3, 2, 3), operation.reg.shape
            assert torch.equal(output_0.cpu(), torch.full((2, 3), expected_0, dtype=global_config.stype)), \
                f'step {step}: input=({bit_0},{bit_1}) output_0={output_0.cpu().tolist()}'
            assert torch.equal(output_1.cpu(), torch.full((2, 3), expected_1, dtype=global_config.stype)), \
                f'step {step}: input=({bit_1},{bit_1}) output_1={output_1.cpu().tolist()}'
            assert operation.timestep_cur == step
            collected.append((output_0.cpu(), output_1.cpu()))

        operation.reset()
        assert operation.timestep_cur == 0
        assert operation.reg.shape == (2, 3), operation.reg.shape
        assert torch.equal(operation.reg.cpu(),
                           torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=global_config.stype))

        # Replaying the same inputs after reset must reproduce the same bits.
        for step, (bit_0, bit_1, _, _) in enumerate(SHUFFLE_D4_SEQUENCE):
            input_0 = torch.full((2, 3), bit_0, dtype=global_config.stype).to(device)
            input_1 = torch.full((2, 3), bit_1, dtype=global_config.stype).to(device)
            output_0, output_1 = operation(input_0, input_1)
            assert torch.equal(output_0.cpu(), collected[step][0]), step
            assert torch.equal(output_1.cpu(), collected[step][1]), step


def test_decorr_correlation():
    """Drive the output SCC of two correlated streams toward 0 while preserving both values."""
    timesteps = 256
    torch.manual_seed(0)
    value_cpu_0 = torch.rand(4, 64).mul(0.9).add(0.05).type(global_config.ntype)
    value_cpu_1 = torch.rand(4, 64).mul(0.9).add(0.05).type(global_config.ntype)
    for device in devices():
        value_0 = value_cpu_0.to(device)
        value_1 = value_cpu_1.to(device)
        input_scc, base_scc, base_0, base_1 = _run_pair(None, value_0, value_1, timesteps, device)
        assert base_scc > 0.99, base_scc

        previous = None
        for depth in [1, 2, 4]:
            operation = decorr({'polarity': 'unipolar', 'depth': depth,
                                        'timestep': timesteps, 'generator': 'sys',
                                        'seed': 7}).to(device)
            _, output_scc, out_0, out_1 = _run_pair(operation, value_0, value_1, timesteps, device)
            error_0 = (out_0 - base_0).abs().max().item()
            error_1 = (out_1 - base_1).abs().max().item()
            print(f'[{device}][depth={depth}] input_scc={input_scc:.4f}, '
                  f'output_scc={output_scc:.4f}, '
                  f'value_error=({error_0:.6f}, {error_1:.6f})')
            # Only bits still stored at the end of the run are lost, at most
            # depth - 1 per stream.
            assert error_0 <= (depth - 1) / timesteps + 1e-6, error_0
            assert error_1 <= (depth - 1) / timesteps + 1e-6, error_1
            if depth == 1:
                # A single position is the pass-through path alone.
                assert output_scc == input_scc, (input_scc, output_scc)
            else:
                assert abs(output_scc) < 0.5, output_scc
                assert abs(output_scc) < abs(input_scc) - 0.5, (input_scc, output_scc)
            if previous is not None:
                assert abs(output_scc) <= abs(previous) + 1e-6, (previous, output_scc)
            previous = output_scc
        assert abs(previous) < 0.15, previous


def test_decorr():
    """Verify decorr preserves the first unipolar stream value across the full range."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_decorr_config()
    test_decorr_known_sequence()
    test_decorr_correlation()
    test_decorr()
    print('Test passed.')

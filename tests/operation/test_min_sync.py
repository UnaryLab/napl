"""Test min_sync streaming; gradients are exempt because it has no parameters."""

import torch

from napl.sim.base import global_config
from napl.sim.operation import decode, encode, min_sync, sync
from napl.utils._shared_test import devices, streaming_suite


# Fig. 5b of the DATE 2018 paper: a synchronizer feeding an AND gate. Each row is
# (input_0, input_1, sync_0, sync_1, output), with the synchronizer columns taken
# from the Fig. 3a depth-1 walk and the output column their AND.
SEQUENCE_D1 = [
    (1, 0, 0, 0, 0),  # S1 -> S0, save the unpaired input_0 bit
    (0, 1, 1, 1, 1),  # S0 -> S1, pair the saved input_0 bit
    (0, 1, 0, 0, 0),  # S1 -> S2, save the unpaired input_1 bit
    (0, 1, 0, 1, 0),  # S2 -> S2, no room left, pass through
    (1, 0, 1, 1, 1),  # S2 -> S1, pair the saved input_1 bit
    (1, 1, 1, 1, 1),  # S1 -> S1, inputs agree
    (0, 0, 0, 0, 0),  # S1 -> S1, inputs agree
    (1, 0, 0, 0, 0),  # S1 -> S0, save the unpaired input_0 bit
    (1, 0, 1, 0, 0),  # S0 -> S0, no room left, pass through
]

# Accuracy of the synchronized minimum against min(p_0, p_1); a bare AND gate or
# an OR gate after the synchronizer both land above 0.1 on the same inputs.
ERROR_BOUND = 0.02


def make_operation(polarity, _timestep, _device):
    return min_sync({'polarity': polarity, 'depth': 1})


def make_values(_polarity):
    first = torch.linspace(0.0, 1.0, 128)
    second = torch.linspace(1.0, 0.0, 128)
    return first, second


def make_performance_values(_polarity):
    first = torch.linspace(0.0, 1.0, 131072)
    second = torch.linspace(1.0, 0.0, 131072)
    return first, second


def analytic_reference(values, _polarity):
    return torch.minimum(values[0], values[1])


def known_answer_case(_polarity):
    values = (torch.tensor([0.0, 1.0, 0.5]), torch.tensor([1.0, 0.0, 0.25]))
    return values, torch.tensor([0.0, 0.0, 0.25]), 2.0 / 256


CONFIG = {
    # min_sync is unipolar only: the synchronizer pairs ones and zeros, which
    # has no bipolar meaning.
    'polarities': ['unipolar'],
    'tolerance_scale': 0.2,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
}


def _run(operation, value_0, value_1, timesteps, device):
    codec = {'polarity': 'unipolar', 'timestep': timesteps, 'generator': 'sobol'}
    encoder_0 = encode({**codec, 'dim': 1}).to(device)
    encoder_1 = encode({**codec, 'dim': 3}).to(device)
    decoder = decode({**codec, 'dim': 1}).to(device)
    for _ in range(timesteps):
        decoder(operation(encoder_0(value_0), encoder_1(value_1)))
    return decoder.spike_value


def _rmse(result, reference):
    return (result - reference).pow(2).mean().sqrt().item()


def test_min_sync_config():
    """Reject a bipolar polarity, missing keys, unknown keys, and an invalid depth."""
    depth_message = 'Invalid depth: <{}>; legal values: an integer of at least 1.'
    for config, message in [
            ({'polarity': 'bipolar', 'depth': 1},
             'Invalid polarity: <bipolar>; min_sync supports unipolar only.'),
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
            min_sync(config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'min_sync accepted the invalid config <{config}>.')


def test_min_sync_known_sequence():
    """Reproduce the Fig. 5b depth-1 composition step by step, then reset and replay it."""
    for device in devices():
        operation = min_sync({'polarity': 'unipolar', 'depth': 1}).to(device)
        for run in ('first', 'replay'):
            for step, (bit_0, bit_1, _, _, expected) in enumerate(SEQUENCE_D1, start=1):
                # Both rows carry the same bits, so the per-element machines stay in step.
                input_0 = torch.full((2, 3), bit_0, dtype=global_config.stype).to(device)
                input_1 = torch.full((2, 3), bit_1, dtype=global_config.stype).to(device)
                output = operation(input_0, input_1)
                assert output.shape == (2, 3), output.shape
                assert operation.sync.cnt.shape == (2, 3), operation.sync.cnt.shape
                assert torch.equal(output.cpu(), torch.full((2, 3), expected, dtype=global_config.stype)), \
                    f'{run} step {step}: input=({bit_0},{bit_1}) output={output.cpu().tolist()}'
                assert operation.timestep_cur == step
                assert operation.sync.timestep_cur == step
            operation.reset()
            assert operation.timestep_cur == 0
            assert operation.sync.timestep_cur == 0
            assert operation.sync.cnt.numel() == 1 and operation.sync.cnt.item() == 0


def test_min_sync_accuracy():
    """Match min(p_0, p_1) at depths 1, 2, and 4, where a wrong composition does not."""
    timesteps = 256
    torch.manual_seed(0)
    value_cpu_0 = torch.rand(4, 64).mul(0.9).add(0.05).type(global_config.ntype)
    value_cpu_1 = torch.rand(4, 64).mul(0.9).add(0.05).type(global_config.ntype)
    for device in devices():
        value_0 = value_cpu_0.to(device)
        value_1 = value_cpu_1.to(device)
        reference = torch.minimum(value_0, value_1)

        previous = None
        for depth in [1, 2, 4]:
            operation = min_sync({'polarity': 'unipolar', 'depth': depth}).to(device)
            error = _rmse(_run(operation, value_0, value_1, timesteps, device), reference)
            print(f'[{device}][depth={depth}] min_sync rmse={error:.6f}, bound={ERROR_BOUND}')
            assert error < ERROR_BOUND, error
            if previous is not None:
                # Bits still saved when the run ends are never emitted, which
                # costs at most depth / timesteps and grows with the depth.
                assert error <= previous + depth / timesteps, (previous, error)
            previous = error

        # A bare AND gate returns p_0 * p_1 on uncorrelated streams.
        def bare_and(spike_0, spike_1):
            return (torch.ne(spike_0, 0) & torch.ne(spike_1, 0)).type(global_config.stype)

        # Swapping the AND gate for an OR gate returns the maximum instead.
        synchronizer = sync({'polarity': 'unipolar', 'depth': 1}).to(device)

        def sync_or(spike_0, spike_1):
            sync_0, sync_1 = synchronizer(spike_0, spike_1)
            return (torch.ne(sync_0, 0) | torch.ne(sync_1, 0)).type(global_config.stype)

        for name, wrong in [('bare_and', bare_and), ('sync_or', sync_or)]:
            error = _rmse(_run(wrong, value_0, value_1, timesteps, device), reference)
            print(f'[{device}] wrong composition {name} rmse={error:.6f}, bound={ERROR_BOUND}')
            assert error > ERROR_BOUND, (name, error)


def test_min_sync():
    """Verify min_sync tracks the analytic minimum across the full unipolar range."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_min_sync_config()
    test_min_sync_known_sequence()
    test_min_sync_accuracy()
    test_min_sync()
    print('Test passed.')

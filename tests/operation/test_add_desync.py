"""Test add_desync streaming; gradients are exempt because it has no parameters."""

import torch

from napl.sim.base import global_config
from napl.sim.operation import add_desync, decode, encode, sync
from napl.utils._shared_test import devices, streaming_suite


# Fig. 5c of the DATE 2018 paper: a desynchronizer feeding an OR gate. Each row
# is (input_0, input_1, desync_0, desync_1, output), walked from the Fig. 3b
# initial state S0 with save depth 1; the output column is the OR of the two
# desynchronizer columns.
SEQUENCE_D1 = [
    (1, 1, 0, 1, 1),  # S0 -> S1, save the paired input_0 one
    (0, 0, 1, 0, 1),  # S1 -> S0, emit the saved input_0 one, saving side flips
    (1, 1, 1, 0, 1),  # S0 -> S3, save the paired input_1 one
    (1, 1, 1, 1, 1),  # S3 -> S3, no room left, pass through
    (0, 0, 0, 1, 1),  # S3 -> S0, emit the saved input_1 one, saving side flips
    (1, 0, 1, 0, 1),  # S0 -> S0, inputs already unpaired
    (0, 1, 0, 1, 1),  # S0 -> S0, inputs already unpaired
    (0, 0, 0, 0, 0),  # S0 -> S0, nothing saved to emit
]

# A strictly saturating pair, p_0 = p_1 = 0.9 over ten steps: input_0 drops its
# one at step 3 and input_1 at step 7. The columns are the same
# (input_0, input_1, desync_0, desync_1, output) as above, walked from S0 with
# save depth 1. Every step emits a one, so the decoded output is 1.0, the
# saturated value of min(1, 1.8). Once the single save slot fills at step 1 the
# machine has no room, so most steps pass both ones through and the two
# desynchronizer columns are one together.
SEQUENCE_SATURATING_D1 = [
    (1, 1, 0, 1, 1),  # S0 -> S1, save the paired input_0 one
    (1, 1, 1, 1, 1),  # S1 -> S1, no room left, pass through
    (0, 1, 0, 1, 1),  # S1 -> S1, inputs already unpaired
    (1, 1, 1, 1, 1),  # S1 -> S1, no room left, pass through
    (1, 1, 1, 1, 1),  # S1 -> S1, no room left, pass through
    (1, 1, 1, 1, 1),  # S1 -> S1, no room left, pass through
    (1, 0, 1, 0, 1),  # S1 -> S1, inputs already unpaired
    (1, 1, 1, 1, 1),  # S1 -> S1, no room left, pass through
    (1, 1, 1, 1, 1),  # S1 -> S1, no room left, pass through
    (1, 1, 1, 1, 1),  # S1 -> S1, no room left, pass through
]

# Accuracy of the desynchronized saturating sum against min(1, p_0 + p_1); a bare
# OR gate or a synchronizer before the OR gate both land above 0.1 on the same
# inputs.
ERROR_BOUND = 0.05


def make_operation(polarity, _timestep, _device):
    return add_desync({'polarity': polarity, 'depth': 1})


def make_values(polarity, count=128):
    # Covered: rate sums p_0 + p_1 from 1/2 to 3/4, a quarter below the
    # saturation boundary at 1, with the same sums under both polarities.
    # Excluded: the band just below and at the boundary. The desynchronizer
    # absorbs a collision (both streams one, arriving at rate p_0 * p_1) by
    # holding a one until a step where both streams are zero (rate
    # (1 - p_0)(1 - p_1)), so the per-step slack is 1 - p_0 - p_1 and vanishes
    # at the boundary. A depth-1 machine holds one spare one, and once the
    # backlog outgrows that slot the dropped ones are a systematic rate loss
    # rather than a stochastic one. test_add_desync_accuracy exercises that
    # band directly, against the wrong-composition discrimination check.
    if polarity == 'bipolar':
        return torch.linspace(-1.0, 0.5, count), torch.linspace(0.0, -1.0, count)
    return torch.linspace(0.0, 0.75, count), torch.linspace(0.5, 0.0, count)


def make_random_perf_values(polarity):
    return make_values(polarity, count=131072)


def analytic_reference(values, polarity):
    # Bipolar reads the same saturating rate sum as min(1, v_0 + v_1 + 1),
    # since p = (v + 1) / 2 turns p_0 + p_1 into (v_0 + v_1) / 2 + 1.
    if polarity == 'bipolar':
        return (values[0] + values[1] + 1.0).clamp(max=1.0)
    return (values[0] + values[1]).clamp(max=1.0)


def known_answer_case(polarity):
    if polarity == 'bipolar':
        values = (torch.tensor([-1.0, 1.0, -0.5]), torch.tensor([-1.0, -1.0, 0.0]))
        return values, torch.tensor([-1.0, 1.0, 0.5])
    values = (torch.tensor([0.0, 1.0, 0.5]), torch.tensor([1.0, 0.0, 0.25]))
    return values, torch.tensor([1.0, 1.0, 0.75])


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
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


def test_add_desync_config():
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
            add_desync(config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'add_desync accepted the invalid config <{config}>.')


def test_add_desync_known_sequence():
    """Reproduce the Fig. 5c and the saturating depth-1 walks step by step under both polarities, then reset and replay each."""
    for device in devices():
        # Polarity only reinterprets the rate, so both polarities emit these bits.
        for polarity in ('unipolar', 'bipolar'):
            for label, sequence in [('fig5c', SEQUENCE_D1), ('saturating', SEQUENCE_SATURATING_D1)]:
                operation = add_desync({'polarity': polarity, 'depth': 1}).to(device)
                assert operation.polarity_io['output'] == polarity
                for run in ('first', 'replay'):
                    for step, (bit_0, bit_1, _, _, expected) in enumerate(sequence, start=1):
                        # Both rows carry the same bits, so the per-element machines stay in step.
                        input_0 = torch.full((2, 3), bit_0, dtype=global_config.stype).to(device)
                        input_1 = torch.full((2, 3), bit_1, dtype=global_config.stype).to(device)
                        output = operation(input_0, input_1)
                        assert output.shape == (2, 3), output.shape
                        assert operation.desync.cnt.shape == (2, 3), operation.desync.cnt.shape
                        assert torch.equal(output.cpu(), torch.full((2, 3), expected, dtype=global_config.stype)), \
                            f'{polarity} {label} {run} step {step}: input=({bit_0},{bit_1}) output={output.cpu().tolist()}'
                        assert operation.timestep_cur == step
                        assert operation.desync.timestep_cur == step
                    operation.reset()
                    assert operation.timestep_cur == 0
                    assert operation.desync.timestep_cur == 0
                    assert operation.desync.cnt.numel() == 1 and operation.desync.cnt.item() == 0
                    assert operation.desync.side.numel() == 1 and operation.desync.side.item() == 1


def test_add_desync_accuracy():
    """Match min(1, p_0 + p_1) across the saturation boundary at depths 1, 2, and 4, where a bare OR gate and a synchronized OR gate do not; unipolar only, since polarity only reinterprets the same rate and never changes the emitted bits."""
    timesteps = 256
    torch.manual_seed(0)
    value_cpu_0 = torch.rand(4, 64).mul(0.9).add(0.05).type(global_config.ntype)
    value_cpu_1 = torch.rand(4, 64).mul(0.9).add(0.05).type(global_config.ntype)
    for device in devices():
        value_0 = value_cpu_0.to(device)
        value_1 = value_cpu_1.to(device)
        reference = (value_0 + value_1).clamp(max=1.0)

        for depth in [1, 2, 4]:
            operation = add_desync({'polarity': 'unipolar', 'depth': depth}).to(device)
            error = _rmse(_run(operation, value_0, value_1, timesteps, device), reference)
            print(f'[{device}][depth={depth}] add_desync rmse={error:.6f}')

        # A bare OR gate returns p_0 + p_1 - p_0 * p_1 on uncorrelated streams.
        def bare_or(spike_0, spike_1):
            return (torch.ne(spike_0, 0) | torch.ne(spike_1, 0)).type(global_config.stype)

        # Swapping the desynchronizer for a synchronizer maximizes the collisions
        # in the OR gate instead of removing them, giving max(p_0, p_1).
        synchronizer = sync({'polarity': 'unipolar', 'depth': 1}).to(device)

        def sync_or(spike_0, spike_1):
            sync_0, sync_1 = synchronizer(spike_0, spike_1)
            return (torch.ne(sync_0, 0) | torch.ne(sync_1, 0)).type(global_config.stype)

        for name, wrong in [('bare_or', bare_or), ('sync_or', sync_or)]:
            error = _rmse(_run(wrong, value_0, value_1, timesteps, device), reference)
            print(f'[{device}] wrong composition {name} rmse={error:.6f}')
            assert error > ERROR_BOUND, (name, error)


def test_add_desync():
    """Verify add_desync tracks its analytic saturating sum in both polarities over the rate sums make_values covers."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_add_desync_config()
    test_add_desync_known_sequence()
    test_add_desync_accuracy()
    test_add_desync()
    print('Test passed.')

"""Test mul_unibi streaming; gradients are exempt because it has no parameters.

mul_unibi is a user-designed mixed-polarity multiplier with no UnarySim
counterpart, so the analytic identity ``v_z = p_u * v_b`` is the only oracle.
"""

import torch

from napl.sim.base import global_config
from napl.sim.operation import decode, encode, mul_unibi
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256


def _spike(bit, shape, device):
    return torch.full(shape, bit, dtype=global_config.stype, device=device)


def make_operation(_polarity, _timestep, _device):
    return mul_unibi()


def make_values(_polarity):
    # A grid spanning the full legal range of both operands.
    value_u = torch.linspace(0.0, 1.0, 16).reshape(16, 1).expand(16, 16).contiguous()
    value_b = torch.linspace(-1.0, 1.0, 16).reshape(1, 16).expand(16, 16).contiguous()
    return value_u, value_b


def make_random_perf_values(_polarity):
    value_u = torch.linspace(0.0, 1.0, 131072)
    value_b = torch.linspace(-1.0, 1.0, 131072)
    return value_u, value_b


def analytic_reference(values, _polarity):
    return values[0] * values[1]


def known_answer_case(_polarity):
    values = (torch.tensor([1.0, 0.0, 0.5]), torch.tensor([1.0, -1.0, 0.5]))
    return values, torch.tensor([1.0, 0.0, 0.25])


CONFIG = {
    # The port polarities are fixed by the operation, so the suite runs once
    # over the bipolar output with a unipolar and a bipolar input.
    'polarities': ['bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'input_polarities': ['unipolar', 'bipolar'],
    'output_polarity': 'bipolar',
    'encoder_dims': [1, 2],
    'timesteps': TIMESTEPS,
}


def _sweep(timesteps, device):
    """Decode the mixed-polarity product of a rank-2 operand grid."""
    value_u, value_b = make_values('bipolar')
    value_u = value_u.to(device)
    value_b = value_b.to(device)
    encoder_u = encode({'polarity': 'unipolar', 'timestep': timesteps,
                        'generator': 'sobol', 'dim': 1}).to(device)
    encoder_b = encode({'polarity': 'bipolar', 'timestep': timesteps,
                        'generator': 'sobol', 'dim': 2}).to(device)
    decoder = decode({'polarity': 'bipolar', 'timestep': timesteps}).to(device)
    operation = mul_unibi().to(device)
    for _ in range(timesteps):
        decoder(operation(encoder_u(value_u), encoder_b(value_b)))
    result = decoder.spike_value
    return (result - value_u * value_b).abs().max().item()


def test_mul_unibi_config():
    """Reject an unknown key and an invalid polarity, and accept the empty config."""
    for config, message in [
            ({'width': 3},
             "Unknown key <width> in the input configuration; accepted keys: <['name', 'polarity']>."),
            ({'timestep': 256},
             "Unknown key <timestep> in the input configuration; accepted keys: <['name', 'polarity']>."),
            ({'polarity': 'mixed'},
             "Invalid polarity: <mixed>; legal values: <['unipolar', 'bipolar']>.")]:
        try:
            mul_unibi(config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'mul_unibi accepted the invalid config <{config}>.')
    assert mul_unibi({}).polarity is None
    assert mul_unibi({'name': 'probe'}).name == 'probe'


def test_mul_unibi_contract():
    """Report rate coding on every port, mixed polarity, zero delay, and a per-element divider buffer."""
    operation = mul_unibi()
    assert operation.streaming is True
    assert operation.encoding_io == {'input_u': 'rc', 'input_b': 'rc', 'output': 'rc'}
    assert operation.polarity_io == {'input_u': 'unipolar', 'input_b': 'bipolar',
                                     'output': 'bipolar'}
    assert operation.correlation_i == {('input_u', 'input_b'): 'zero'}
    assert operation.hw.pp_delay == 0
    assert 'state' in dict(operation.named_buffers())
    assert operation.state.shape == (1,)
    assert operation.state.dtype == torch.int8
    assert operation.stability_flux == 1.0
    # The divider flip-flop is per element, so it takes the input shape on the
    # first call and toggles on every element where input_u is low.
    operation(_spike(0, (2, 3), 'cpu'), _spike(1, (2, 3), 'cpu'))
    assert operation.state.shape == (2, 3)
    assert torch.equal(operation.state, torch.ones((2, 3), dtype=torch.int8))
    operation.reset()
    assert operation.timestep_cur == 0
    assert operation.state.shape == (1,)
    assert operation.state.item() == 0


def test_mul_unibi_known_answer():
    """Pass a bipolar stream through bit-exactly at u=1 and emit every second event at u=0."""
    shape = (2, 3)
    for device in devices():
        encoder_b = encode({'polarity': 'bipolar', 'timestep': TIMESTEPS,
                            'generator': 'sobol', 'dim': 2}).to(device)
        value_b = torch.full(shape, 0.25, dtype=global_config.ntype, device=device)

        # u constant one: the AND leg passes input_b through unchanged.
        operation = mul_unibi().to(device)
        for step in range(1, TIMESTEPS + 1):
            spike_b = encoder_b(value_b)
            output = operation(_spike(1, shape, device), spike_b)
            assert output.shape == shape, output.shape
            assert torch.equal(output, spike_b), step
            assert operation.timestep_cur == step
        assert operation.state.shape == shape, operation.state.shape

        # u constant zero: every timestep is an event, so the divider emits on
        # every second one, 0, 1, 0, 1, ...
        operation.reset()
        encoder_b.reset()
        assert operation.timestep_cur == 0
        assert operation.state.shape == (1,), operation.state.shape
        assert operation.state.item() == 0
        decoder = decode({'polarity': 'bipolar', 'timestep': TIMESTEPS}).to(device)
        trace = []
        for step in range(1, TIMESTEPS + 1):
            output = operation(_spike(0, shape, device), encoder_b(value_b))
            assert torch.equal(output, _spike((step - 1) % 2, shape, device)), step
            trace.append(output.clone())
            decoder(output)
        # An even event count makes the divider rate exactly one half.
        assert torch.equal(decoder.spike_value,
                           torch.zeros(shape, dtype=global_config.ntype, device=device))

        # An odd event count leaves half an event in the flip-flop, biasing the
        # decoded output by exactly -1/N.
        operation.reset()
        odd_steps = TIMESTEPS - 1
        odd_decoder = decode({'polarity': 'bipolar', 'timestep': odd_steps}).to(device)
        for _ in range(odd_steps):
            odd_decoder(operation(_spike(0, shape, device), _spike(1, shape, device)))
        assert torch.equal(
            odd_decoder.spike_count,
            torch.full(shape, odd_steps // 2, dtype=global_config.ntype, device=device))
        torch.testing.assert_close(
            odd_decoder.spike_value,
            torch.full(shape, -1.0 / odd_steps, dtype=global_config.ntype, device=device),
            atol=1e-7, rtol=0)

        # Reset and replay reproduce the divider output bit for bit.
        operation.reset()
        assert operation.state.item() == 0
        for step in range(1, TIMESTEPS + 1):
            output = operation(_spike(0, shape, device), _spike(1, shape, device))
            assert torch.equal(output, trace[step - 1]), step


def test_mul_unibi_analytic():
    """Match p_u * v_b over a decorrelated Sobol grid at even and odd run lengths."""
    for device in devices():
        for timesteps in [256, 255]:
            error = _sweep(timesteps, device)
            print(f'[{device}] N={timesteps} ({"even" if timesteps % 2 == 0 else "odd"}), '
                  f'max_error={error:.6f}')


def test_mul_unibi_period2():
    """Hold the identity when input_u carries period-2 structure.

    The event-driven divider counts events, not timesteps, so it is exact on
    any spacing of the zeros in ``input_u``. A timestep-clocked halver, whose
    /2 leg toggles every timestep, fails this case because the two period-2
    streams below place every ``u = 0`` timestep on one toggle phase and shift
    the output by about ``+/- (1 - p_u) = 0.5``.

    With ``p_u = 0.5`` every label sees 128 events over 256 timesteps, so the
    divider emits 64 and the decoded value is exactly ``2 * 64 / 256 - 1``.
    """
    shape = (2, 3)
    for device in devices():
        measured = {}
        for label in ['aligned', 'antialigned', 'sobol']:
            encoder_u = encode({'polarity': 'unipolar', 'timestep': TIMESTEPS,
                                'generator': 'sobol', 'dim': 1}).to(device)
            encoder_b = encode({'polarity': 'bipolar', 'timestep': TIMESTEPS,
                                'generator': 'sobol', 'dim': 2}).to(device)
            decoder = decode({'polarity': 'bipolar', 'timestep': TIMESTEPS}).to(device)
            operation = mul_unibi().to(device)
            value_u = torch.full(shape, 0.5, dtype=global_config.ntype, device=device)
            # input_b = -1 is an all-zero stream, so the AND leg contributes
            # nothing and every deviation comes from the divider leg.
            value_b = torch.full(shape, -1.0, dtype=global_config.ntype, device=device)
            for step in range(1, TIMESTEPS + 1):
                if label == 'sobol':
                    spike_u = encoder_u(value_u)
                else:
                    # The two complementary period-2 streams that a
                    # timestep-clocked halver gets wrong.
                    odd = step % 2
                    spike_u = _spike(odd if label == 'aligned' else 1 - odd, shape, device)
                decoder(operation(spike_u, encoder_b(value_b)))
            measured[label] = decoder.spike_value.mean().item()
        expected = 0.5 * -1.0
        print(f'[{device}] period-2 robustness, expected v_z={expected:.6f}: '
              f'aligned={measured["aligned"]:.6f}, '
              f'antialigned={measured["antialigned"]:.6f}, '
              f'sobol={measured["sobol"]:.6f}')
        for label, value in measured.items():
            assert value == expected, (label, measured)


def test_mul_unibi():
    """Verify mul_unibi against the analytic identity, with reset, replay, and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_mul_unibi_config()
    test_mul_unibi_contract()
    test_mul_unibi_known_answer()
    test_mul_unibi_analytic()
    test_mul_unibi_period2()
    test_mul_unibi()
    print('Test passed.')

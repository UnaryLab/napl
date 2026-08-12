"""Test mul_unibi_mux streaming; gradients are exempt because it has no parameters.

mul_unibi_mux is a user-designed mixed-polarity multiplier with no UnarySim
counterpart, so the analytic identity ``v_z = p_u * v_b`` is the only oracle.
"""

import torch

from napl.sim.base import global_config
from napl.sim.operation import decode, encode, mul_unibi_mux
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256


def _spike(bit, shape, device):
    return torch.full(shape, bit, dtype=global_config.stype, device=device)


def make_operation(_polarity, _timestep, _device):
    return mul_unibi_mux()


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
    operation = mul_unibi_mux().to(device)
    for _ in range(timesteps):
        decoder(operation(encoder_u(value_u), encoder_b(value_b)))
    result = decoder.spike_value
    return (result - value_u * value_b).abs().max().item()


def test_mul_unibi_mux_config():
    """Reject an unknown key and an invalid polarity, and accept the empty config."""
    for config, message in [
            ({'width': 3},
             "Unknown key <width> in the input configuration; accepted keys: <['name', 'polarity']>."),
            ({'timestep': 256},
             "Unknown key <timestep> in the input configuration; accepted keys: <['name', 'polarity']>."),
            ({'polarity': 'mixed'},
             "Invalid polarity: <mixed>; legal values: <['unipolar', 'bipolar']>.")]:
        try:
            mul_unibi_mux(config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'mul_unibi_mux accepted the invalid config <{config}>.')
    assert mul_unibi_mux({}).polarity is None
    assert mul_unibi_mux({'name': 'probe'}).name == 'probe'


def test_mul_unibi_mux_contract():
    """Report rate coding on every port, mixed polarity, zero delay, and a scalar toggle buffer."""
    operation = mul_unibi_mux()
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
    # The toggle state is scalar and stays scalar, so it broadcasts to any shape.
    operation(_spike(1, (2, 3), 'cpu'), _spike(1, (2, 3), 'cpu'))
    assert operation.state.shape == (1,)
    # This is also the only check on the toggle rate: a half-rate toggle keeps the
    # mean at 0.5, so the decoded product moves too little for any rmse gate to see.
    assert operation.state.item() == 1
    operation.reset()
    assert operation.timestep_cur == 0
    assert operation.state.item() == 0


def test_mul_unibi_mux_known_answer():
    """Pass a bipolar stream through bit-exactly at u=1 and emit the toggle stream at u=0."""
    shape = (2, 3)
    for device in devices():
        encoder_b = encode({'polarity': 'bipolar', 'timestep': TIMESTEPS,
                            'generator': 'sobol', 'dim': 2}).to(device)
        value_b = torch.full(shape, 0.25, dtype=global_config.ntype, device=device)

        # u constant one: the select leg passes input_b through unchanged.
        operation = mul_unibi_mux().to(device)
        for step in range(1, TIMESTEPS + 1):
            spike_b = encoder_b(value_b)
            output = operation(_spike(1, shape, device), spike_b)
            assert output.shape == shape, output.shape
            assert torch.equal(output, spike_b), step
            assert operation.timestep_cur == step
        assert operation.state.shape == (1,), operation.state.shape

        # u constant zero: the output is the toggle stream, 0, 1, 0, 1, ...
        operation.reset()
        encoder_b.reset()
        assert operation.timestep_cur == 0
        assert operation.state.item() == 0
        decoder = decode({'polarity': 'bipolar', 'timestep': TIMESTEPS}).to(device)
        trace = []
        for step in range(1, TIMESTEPS + 1):
            output = operation(_spike(0, shape, device), encoder_b(value_b))
            assert torch.equal(output, _spike((step - 1) % 2, shape, device)), step
            trace.append(output.clone())
            decoder(output)
        # An even timestep count makes the toggle rate exactly one half.
        assert torch.equal(decoder.spike_value,
                           torch.zeros(shape, dtype=global_config.ntype, device=device))

        # An odd run count leaves the toggle at one, so reset has to clear it.
        operation.reset()
        for _ in range(TIMESTEPS - 1):
            operation(_spike(0, shape, device), _spike(1, shape, device))
        assert operation.state.item() == 1, operation.state
        operation.reset()
        assert operation.state.item() == 0, operation.state

        # Reset and replay reproduce the toggle output bit for bit.
        operation.reset()
        assert operation.state.item() == 0
        for step in range(1, TIMESTEPS + 1):
            output = operation(_spike(0, shape, device), _spike(1, shape, device))
            assert torch.equal(output, trace[step - 1]), step


def test_mul_unibi_mux_analytic():
    """Match p_u * v_b over a decorrelated Sobol grid at even and odd run lengths."""
    for device in devices():
        for timesteps in [256, 255]:
            error = _sweep(timesteps, device)
            print(f'[{device}] N={timesteps} ({"even" if timesteps % 2 == 0 else "odd"}), '
                  f'max_error={error:.6f}')


def test_mul_unibi_mux_period2():
    """Pin the exact parity bias when input_u carries period-2 structure.

    The toggle stream is clocked on every timestep, so an ``input_u`` locked to
    one timestep parity meets a fixed toggle phase and the decoded output is
    exact, not statistical. With ``p_u = 0.5`` and ``input_b = -1``, an all-zero
    stream that makes the select leg contribute nothing, the aligned stream puts
    every ``u = 0`` timestep on the toggle's one phase and decodes to
    ``2 * 128 / 256 - 1 = 0.0``, while the antialigned stream puts them all on
    the zero phase and decodes to ``-1.0``. Both are half a unit away from
    ``p_u * v_b = -0.5``, the bias the class docstring documents.
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
            operation = mul_unibi_mux().to(device)
            value_u = torch.full(shape, 0.5, dtype=global_config.ntype, device=device)
            value_b = torch.full(shape, -1.0, dtype=global_config.ntype, device=device)
            for step in range(1, TIMESTEPS + 1):
                if label == 'sobol':
                    spike_u = encoder_u(value_u)
                else:
                    odd = step % 2
                    spike_u = _spike(odd if label == 'aligned' else 1 - odd, shape, device)
                decoder(operation(spike_u, encoder_b(value_b)))
            measured[label] = decoder.spike_value.mean().item()
        expected = 0.5 * -1.0
        print(f'[{device}] period-2 parity lock, p_u*v_b={expected:.6f}: '
              f'aligned={measured["aligned"]:.6f}, '
              f'antialigned={measured["antialigned"]:.6f}, '
              f'sobol={measured["sobol"]:.6f}')
        assert measured['aligned'] == 0.0, measured
        assert measured['antialigned'] == -1.0, measured


def test_mul_unibi_mux():
    """Verify mul_unibi_mux against the analytic identity, with reset, replay, and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_mul_unibi_mux_config()
    test_mul_unibi_mux_contract()
    test_mul_unibi_mux_known_answer()
    test_mul_unibi_mux_analytic()
    test_mul_unibi_mux_period2()
    test_mul_unibi_mux()
    print('Test passed.')

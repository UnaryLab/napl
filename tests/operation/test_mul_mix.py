"""Test mul_mix streaming; gradients are exempt because it has no parameters.

mul_mix is a user-designed mixed-polarity multiplier with no UnarySim
counterpart, so the analytic identity ``v_z = p_u * v_b`` is the only oracle.
"""

import math

import torch

from napl.sim.base import global_config
from napl.sim.operation import decode, encode, mul_mix
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256


def bound(timesteps):
    """Coarse per-element ceiling on the output error in the bipolar domain.

    This is a working ceiling, not a proof: the ``6/N`` part is derived, the
    ``2/sqrt(N)`` part is a chosen coverage allowance for residual correlation.

    Over ``N`` timesteps the output count splits into the two disjoint gate
    legs, ``C_and = #{u=1 and b=1}`` and ``C_div = floor(Z / 2)`` with
    ``Z = #{u=0}`` the divider event count, and the decoded value is
    ``2 * (C_and + C_div) / N - 1``. Three error terms separate it from
    ``p_u * (2 * p_b - 1)``, each doubled by the bipolar decode:

    1. Encoder quantization (derived). A Sobol comparator stream of value ``p``
       holds a count within one of ``N * p``, so ``|p_hat - p| <= 1/N``. The
       product term carries ``2/N`` of that and the ``(1 - p_hat_u)/2`` term
       ``1/(2N)``, giving ``5/N`` after doubling.
    2. Residual correlation on the AND leg (chosen, not derived). Distinct
       Sobol dimensions leave a nonzero sample correlation. The operands are
       deterministic Sobol sequences, not random draws, so no sigma argument
       applies and no tight closed form is claimed here; the factor 2 in
       ``2/sqrt(N)`` is a coverage multiplier picked over the ``1/sqrt(N)``
       scale that this deviation is observed to follow.
    3. Divider truncation (derived). The divider emits exactly
       ``floor(Z / 2)``, so it falls short of ``Z / 2`` by half a count when
       ``Z`` is odd and by nothing when ``Z`` is even: at most ``1/(2N)`` in
       rate, giving ``1/N`` after doubling, always negative. The odd-run bias
       at ``u == 0`` is this term at its extreme, ``-1/N``.

    Total: ``2/sqrt(N) + 6/N``. The divider is exact on events regardless of
    how they are spaced, so term 3 holds for any structure in ``u``; term 2
    still assumes ``u`` and ``b`` are decorrelated.
    """
    return 2.0 / math.sqrt(timesteps) + 6.0 / timesteps


def _spike(bit, shape, device):
    return torch.full(shape, bit, dtype=global_config.stype, device=device)


def make_operation(_polarity, _timestep, _device):
    return mul_mix()


def make_values(_polarity):
    # A grid spanning the full legal range of both operands.
    value_u = torch.linspace(0.0, 1.0, 16).reshape(16, 1).expand(16, 16).contiguous()
    value_b = torch.linspace(-1.0, 1.0, 16).reshape(1, 16).expand(16, 16).contiguous()
    return value_u, value_b


def make_performance_values(_polarity):
    value_u = torch.linspace(0.0, 1.0, 131072)
    value_b = torch.linspace(-1.0, 1.0, 131072)
    return value_u, value_b


def analytic_reference(values, _polarity):
    return values[0] * values[1]


def known_answer_case(_polarity):
    values = (torch.tensor([1.0, 0.0, 0.5]), torch.tensor([1.0, -1.0, 0.5]))
    return values, torch.tensor([1.0, 0.0, 0.25]), bound(TIMESTEPS)


CONFIG = {
    # The port polarities are fixed by the operation, so the suite runs once
    # over the bipolar output with a unipolar and a bipolar input.
    'polarities': ['bipolar'],
    # Gate set from the suite's own statistic, not from bound(): the suite rmse
    # over the 16x16 operand grid measures 0.005413 at N=256 on cpu and mps, so
    # 0.35/sqrt(N) = 0.021875 sits at 4x the measured baseline. Four is enough
    # slack for seed and device jitter while still failing any mutant that
    # perturbs the emitted spike count by a few percent (zeroing the output on
    # every 16th timestep measures rmse 0.0626, 2.86x over this gate; dropping
    # every 16th emitted spike measures 0.060864).
    'tolerance_scale': 0.35,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
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
    operation = mul_mix().to(device)
    for _ in range(timesteps):
        decoder(operation(encoder_u(value_u), encoder_b(value_b)))
    result = decoder.spike_value
    return (result - value_u * value_b).abs().max().item()


def test_mul_mix_config():
    """Reject an unknown key and an invalid polarity, and accept the empty config."""
    for config, message in [
            ({'width': 3},
             "Unknown key <width> in the input configuration; accepted keys: <['name', 'polarity']>."),
            ({'timestep': 256},
             "Unknown key <timestep> in the input configuration; accepted keys: <['name', 'polarity']>."),
            ({'polarity': 'mixed'},
             "Invalid polarity: <mixed>; legal values: <['unipolar', 'bipolar']>.")]:
        try:
            mul_mix(config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'mul_mix accepted the invalid config <{config}>.')
    assert mul_mix({}).polarity is None
    assert mul_mix({'name': 'probe'}).name == 'probe'


def test_mul_mix_known_answer():
    """Pass a bipolar stream through bit-exactly at u=1 and emit every second event at u=0."""
    shape = (2, 3)
    for device in devices():
        encoder_b = encode({'polarity': 'bipolar', 'timestep': TIMESTEPS,
                            'generator': 'sobol', 'dim': 2}).to(device)
        value_b = torch.full(shape, 0.25, dtype=global_config.ntype, device=device)

        # u constant one: the AND leg passes input_b through unchanged.
        operation = mul_mix().to(device)
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


def test_mul_mix_analytic():
    """Match p_u * v_b over a decorrelated Sobol grid at even and odd run lengths."""
    for device in devices():
        for timesteps in [256, 255]:
            error = _sweep(timesteps, device)
            limit = bound(timesteps)
            print(f'[{device}] N={timesteps} ({"even" if timesteps % 2 == 0 else "odd"}), '
                  f'max_error={error:.6f}, bound={limit:.6f}')
            assert error <= limit, (device, timesteps, error, limit)


def test_mul_mix_period2():
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
            operation = mul_mix().to(device)
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


def test_mul_mix():
    """Verify mul_mix against the analytic identity, with reset, replay, and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_mul_mix_config()
    test_mul_mix_known_answer()
    test_mul_mix_analytic()
    test_mul_mix_period2()
    test_mul_mix()
    print('Test passed.')

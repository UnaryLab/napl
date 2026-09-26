import torch

from napl.sim.base import global_config
from napl.sim.metric import correlation
from napl.sim.operation import encode, pow_delay
from napl.utils._shared_test import devices, streaming_suite


N = 3
TIMESTEPS = 256


def _operation_config(polarity, n=N, depth=1):
    return {'polarity': polarity, 'n': n, 'depth': depth}


def make_operation(polarity, _timestep, _device):
    return pow_delay(_operation_config(polarity))


def make_values(polarity):
    """Cover the full admitted span; delayed factors impose no narrower domain."""
    generator = torch.Generator().manual_seed(
        2217 + int(polarity == 'bipolar')
    )
    values = torch.rand(128, generator=generator)
    if polarity == 'bipolar':
        values = 2 * values - 1
    return (values,)


def analytic_reference(values, _polarity):
    return values[0].pow(N)


def known_answer_case(polarity):
    values = (torch.tensor([0.0, 0.5, 1.0]) if polarity == 'unipolar'
              else torch.tensor([-1.0, 0.0, 1.0]))
    return (values,), values.pow(N)


def make_random_perf_values(polarity):
    generator = torch.Generator().manual_seed(
        2218 + int(polarity == 'bipolar')
    )
    values = torch.rand(131072, generator=generator)
    if polarity == 'bipolar':
        values = 2 * values - 1
    return (values,)


def _validation_checks():
    """Reject non-integer, too-small, and too-large powers."""
    invalid = [
        (1, 'Invalid n: <1>; legal values: an integer from 2 to 8.'),
        (1.5, 'Invalid n: <1.5>; legal values: an integer from 2 to 8.'),
        (True, 'Invalid n: <True>; legal values: an integer from 2 to 8.'),
        (pow_delay.N_MAX + 1,
         'Invalid n: <9>; legal values: an integer from 2 to 8.'),
    ]
    for n, message in invalid:
        try:
            pow_delay({'polarity': 'unipolar', 'n': n})
        except AssertionError as error:
            assert str(error) == message
        else:
            raise AssertionError(f'pow_delay accepted illegal n={n}')

    try:
        pow_delay({'polarity': 'nonesuch', 'n': 3})
    except AssertionError as error:
        assert str(error) == "Invalid polarity: <nonesuch>; legal values: <['unipolar', 'bipolar']>."
    else:
        raise AssertionError('pow_delay accepted an invalid polarity')

    invalid_depth = [
        (0, 'Invalid depth: <0>; legal values: an integer of at least 1.'),
        (-1, 'Invalid depth: <-1>; legal values: an integer of at least 1.'),
        (1.5, 'Invalid depth: <1.5>; legal values: an integer of at least 1.'),
        (True, 'Invalid depth: <True>; legal values: an integer of at least 1.'),
    ]
    for depth, message in invalid_depth:
        try:
            pow_delay({'polarity': 'unipolar', 'n': 3, 'depth': depth})
        except AssertionError as error:
            assert str(error) == message
        else:
            raise AssertionError(f'pow_delay accepted illegal depth={depth}')


def _bare_trace(stream, polarity, n, depth):
    """Evaluate delayed-factor products without using pow_delay or delay."""
    history = [torch.zeros_like(stream[0]) for _ in range((n - 1) * depth)]
    trace = []
    for spike in stream:
        factors = [spike]
        factors.extend(history[index * depth - 1] for index in range(1, n))
        output = factors[0].type(torch.int8)
        for factor in factors[1:]:
            factor = factor.type(torch.int8)
            output = (output & factor if polarity == 'unipolar'
                      else 1 - (output ^ factor))
        trace.append(output.type(global_config.stype))
        history = [spike.detach().clone(), *history[:-1]]
    return trace


def _structural_checks():
    """Pin delayed factors, reset replay, state shape, and zero-cycle latency."""
    generator = torch.Generator().manual_seed(7319)
    stream_cpu = torch.randint(
        0, 2, (64, 3, 2), generator=generator, dtype=torch.int8,
    ).type(global_config.stype)

    for device in devices():
        stream = stream_cpu.to(device)
        for polarity in ['unipolar', 'bipolar']:
            operation = pow_delay(
                _operation_config(polarity, n=4, depth=3)
            ).to(device)
            expected = _bare_trace(stream, polarity, n=4, depth=3)
            first_trace = [operation(spike) for spike in stream]

            assert operation.hw.pp_delay == 0
            assert operation.internal_encode == 'none'
            assert operation.correlation_o == {}
            assert len(operation.delays) == 3
            assert len(operation.multipliers) == 3
            assert all(output.shape == stream.shape[1:] for output in first_trace)
            assert all(stage.reg.shape == (3, *stream.shape[1:])
                       for stage in operation.delays)
            assert all(torch.equal(output, reference)
                       for output, reference in zip(first_trace, expected))

            operation.reset()
            assert operation.timestep_cur == 0
            assert all(stage.reg.shape == (3,) and stage.head.shape == ()
                       for stage in operation.delays)
            replay = [operation(spike) for spike in stream]
            assert all(torch.equal(first, second)
                       for first, second in zip(first_trace, replay))


def _square_equivalence_checks():
    """Pin n=2 to the independent delayed-square trace at the same depth."""
    generator = torch.Generator().manual_seed(7391)
    stream_cpu = torch.randint(
        0, 2, (64, 2, 3), generator=generator, dtype=torch.int8,
    ).type(global_config.stype)
    for device in devices():
        stream = stream_cpu.to(device)
        for polarity in ['unipolar', 'bipolar']:
            operation = pow_delay(
                _operation_config(polarity, n=2, depth=3)
            ).to(device)
            expected = _bare_trace(stream, polarity, n=2, depth=3)
            output = [operation(spike) for spike in stream]
            assert all(torch.equal(actual, reference)
                       for actual, reference in zip(output, expected))


def _dead_zone_checks():
    """Pin and print each T=256 Sobol zero-output boundary and next point."""
    timestep = 256
    cases = {
        3: (0.50, 0.51, 1),
        4: (0.51, 0.52, 1),
        5: (0.75, 0.76, 5),
        6: (0.75, 0.76, 2),
        7: (0.75, 0.76, 1),
        8: (0.76, 0.77, 3),
    }
    for device in devices():
        for n, (inside, outside, outside_count) in cases.items():
            values = torch.tensor([[inside], [outside]], device=device)
            codec = encode({'polarity': 'unipolar', 'timestep': timestep,
                            'generator': 'sobol', 'dim': 1}).to(device)
            operation = pow_delay({'polarity': 'unipolar', 'n': n}).to(device)
            count = torch.zeros_like(values)
            for _ in range(timestep):
                count.add_(operation(codec(values)))
            expected = torch.tensor(
                [[0.0], [float(outside_count)]], device=device,
            )
            assert torch.equal(count, expected), (
                f'[{device}][n={n}] T=256 Sobol dead-zone boundary changed'
            )
            print(
                f'[{device}][sobol][unipolar][n={n}][T=256][depth=1] '
                f'dead_zone_boundary={inside:.2f} first_nonzero={outside:.2f}'
            )


def _measure(values, polarity, n, timestep, generator='sobol', depth=1):
    """Return decoded values and the input/output SCC for one printed row."""
    codec = encode({'polarity': polarity, 'timestep': timestep,
                    'generator': generator, 'dim': 1})
    operation = pow_delay(_operation_config(polarity, n=n, depth=depth))
    metric = correlation()
    count = torch.zeros_like(values)
    for _ in range(timestep):
        spike = codec(values)
        output = operation(spike)
        count.add_(output)
        metric(spike, output)
    decoded = (count / timestep if polarity == 'unipolar'
               else 2 * count / timestep - 1)
    return decoded, metric.correlation.mean().item()


def _sobol_accuracy_report():
    """Print exact per-n Sobol accuracy and the depth design comparison."""
    for polarity, seed in [('unipolar', 2220), ('bipolar', 2221)]:
        values = torch.rand(4096, generator=torch.Generator().manual_seed(seed))
        if polarity == 'bipolar':
            values = 2 * values - 1
        for n in range(2, pow_delay.N_MAX + 1):
            decoded, input_output_scc = _measure(values, polarity, n, 256)
            error = (decoded - values.pow(n)).abs()
            print(
                f'[cpu][sobol][{polarity}][n={n}][T=256][depth=1] '
                f'mean_abs={error.mean().item():.6f} '
                f'max_abs={error.max().item():.6f} '
                f'input_output_scc={input_output_scc:.6f}'
            )

    values = torch.rand(4096, generator=torch.Generator().manual_seed(2220))
    for depth in [1, 2, 3, 5, 8]:
        for timestep in [256, 1024, 4096]:
            decoded, input_output_scc = _measure(
                values, 'unipolar', n=3, timestep=timestep, depth=depth,
            )
            error = (decoded - values.pow(3)).abs()
            print(
                f'[cpu][comparison=sobol-depth][unipolar][n=3]'
                f'[T={timestep}][depth={depth}] '
                f'mean_abs={error.mean().item():.6f} '
                f'max_abs={error.max().item():.6f} '
                f'input_output_scc={input_output_scc:.6f} '
                f'ff_bits={2 * depth} rom_bits=0 gates=2 logic_depth=2'
            )


def _generator_sensitivity_report():
    """Print LFSR mechanism sensitivity, not caller accuracy evidence."""
    values = torch.rand(4096, generator=torch.Generator().manual_seed(2222))
    for n in range(2, pow_delay.N_MAX + 1):
        decoded, input_output_scc = _measure(
            values, 'unipolar', n, 256, generator='lfsr',
        )
        error = (decoded - values.pow(n)).abs()
        print(
            f'[cpu][evidence=generator-sensitivity][lfsr][unipolar]'
            f'[n={n}][T=256][depth=1] '
            f'mean_abs={error.mean().item():.6f} '
            f'max_abs={error.max().item():.6f} '
            f'input_output_scc={input_output_scc:.6f}'
        )


def _extra_checks():
    """Run validation, structural, collapse, and printed limitation checks."""
    _validation_checks()
    _structural_checks()
    _square_equivalence_checks()
    _dead_zone_checks()
    _sobol_accuracy_report()
    _generator_sensitivity_report()


# The independent delayed-factor oracle pins the per-timestep product, catching
# a skipped delay update or a multiplier whose emitted output is bypassed.
CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': TIMESTEPS,
    'extra_checks': _extra_checks,
}


def test_pow_delay():
    """Verify delayed powers for both polarities and pin their known limitation."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_pow_delay()

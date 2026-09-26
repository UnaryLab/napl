import torch

from napl.sim.base import global_config
from napl.sim.metric import correlation
from napl.sim.operation import encode_regen, encode, pow_regen
from napl.utils._shared_test import devices, streaming_suite


N = 3
TIMESTEPS = 256
SEED = 7319


def _operation_config(**updates):
    config = {'polarity': 'unipolar', 'n': N}
    config.update(updates)
    return config


def make_operation(polarity, _timestep, _device):
    return pow_regen(_operation_config(polarity=polarity))


def make_values(_polarity):
    generator = torch.Generator().manual_seed(SEED)
    return (torch.rand((16, 8), generator=generator),)


def make_random_perf_values(_polarity):
    generator = torch.Generator().manual_seed(SEED + 1)
    return (torch.rand((16384, 8), generator=generator),)


def analytic_reference(values, _polarity):
    return values[0].pow(N)


def known_answer_case(_polarity):
    values = torch.tensor([[0.0, 1.0]])
    return (values,), values.pow(N)


def _bare_trace(stream, n, lfsr_width, window, depth):
    """Return the bit-exact topology without constructing pow_regen."""
    maximal_period = 2 ** lfsr_width - 1
    sequence = encode({
        'polarity': 'unipolar', 'timestep': 2 ** lfsr_width,
        'generator': 'lfsr',
    })
    states = sequence.num_seq[:maximal_period]
    states = states.mul(2 ** lfsr_width).round().type(torch.long).to(stream.device)
    insertion = int(torch.nonzero(states.eq(1), as_tuple=False)[0].item()) + 1
    states = torch.cat((states[:insertion], torch.zeros(1, dtype=torch.long,
                                                        device=stream.device),
                        states[insertion:]))
    period = 2 ** lfsr_width
    depth_span = (n - 1) * depth
    window_register = None
    depth_register = None
    outputs = []

    for timestep, input in enumerate(stream):
        if window_register is None:
            window_register = input.unsqueeze(0).repeat(window, *([1] * input.ndim))

        regenerated = states[timestep % period].remainder(window).lt(
            window_register.sum(dim=0)
        ).type(global_config.stype)
        if timestep:
            window_register[timestep % window] = input
        factors = [regenerated]
        if depth_span:
            if depth_register is None:
                depth_register = regenerated.unsqueeze(0).repeat(
                    depth_span, *([1] * regenerated.ndim)
                )
                factors.extend([regenerated] * (n - 1))
            else:
                head = (timestep - 1) % depth_span
                for tap_depth in range(depth, depth_span + 1, depth):
                    index = (head + depth_span - tap_depth) % depth_span
                    factors.append(depth_register[index].clone())
                depth_register[head] = regenerated

        output = factors[0]
        for factor in factors[1:]:
            output = output.type(torch.int8).bitwise_and(factor.type(torch.int8))
        outputs.append(output.type(global_config.stype))

    return torch.stack(outputs)


def _validation_checks():
    """Reject every unsupported polarity and invalid sizing value exactly."""
    invalid = [
        ({'polarity': 'bipolar'},
         'Invalid polarity: <bipolar>; pow_regen supports unipolar only.'),
        ({'n': 1}, 'Invalid n: <1>; legal values: an integer from 2 to 8.'),
        ({'n': 9}, 'Invalid n: <9>; legal values: an integer from 2 to 8.'),
        ({'n': 3.0}, 'Invalid n: <3.0>; legal values: an integer from 2 to 8.'),
        ({'n': True}, 'Invalid n: <True>; legal values: an integer from 2 to 8.'),
        ({'lfsr_width': 2},
         'Invalid lfsr_width: <2>; legal values: an integer from 3 to 8.'),
        ({'lfsr_width': 9},
         'Invalid lfsr_width: <9>; legal values: an integer from 3 to 8.'),
        ({'lfsr_width': 4.0},
         'Invalid lfsr_width: <4.0>; legal values: an integer from 3 to 8.'),
        ({'lfsr_width': True},
         'Invalid lfsr_width: <True>; legal values: an integer from 3 to 8.'),
        ({'window': 1},
         'Invalid window: <1>; legal values: a power-of-two integer from 2 to 8.'),
        ({'window': 3},
         'Invalid window: <3>; legal values: a power-of-two integer from 2 to 8.'),
        ({'window': 16},
         'Invalid window: <16>; legal values: a power-of-two integer from 2 to 8.'),
        ({'window': 4.0},
         'Invalid window: <4.0>; legal values: a power-of-two integer from 2 to 8.'),
        ({'window': True},
         'Invalid window: <True>; legal values: a power-of-two integer from 2 to 8.'),
        ({'depth': 0}, 'Invalid depth: <0>; legal values: an integer of at least 1.'),
        ({'depth': -1}, 'Invalid depth: <-1>; legal values: an integer of at least 1.'),
        ({'depth': 1.0}, 'Invalid depth: <1.0>; legal values: an integer of at least 1.'),
        ({'depth': True}, 'Invalid depth: <True>; legal values: an integer of at least 1.'),
    ]
    for updates, message in invalid:
        try:
            pow_regen(_operation_config(**updates))
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'pow_regen accepted invalid config {updates}')


def _structural_checks():
    """Pin the registered topology, tap depth, comparator, and reset replay."""
    timesteps = 48
    generator = torch.Generator().manual_seed(SEED + 2)
    stream_cpu = torch.randint(
        0, 2, (timesteps, 3, 2), generator=generator, dtype=torch.int8,
    ).type(global_config.stype)
    config = _operation_config(lfsr_width=4, window=4, depth=2)

    for device in devices():
        stream = stream_cpu.to(device)
        operation = pow_regen(config).to(device)
        expected = _bare_trace(stream, N, 4, 4, 2)
        first_trace = torch.stack([operation(spike) for spike in stream])

        # These exact traces are the depth-removal and flipped-comparator bites.
        assert torch.equal(first_trace, expected), (
            f'[{device}] regeneration topology differs from the independent trace'
        )
        assert first_trace.shape == stream.shape
        assert operation.regenerator.window_register.shape == (4, *stream.shape[1:])
        assert operation.depth_register.shape == (4, *stream.shape[1:])
        assert operation.regenerator.lfsr_index.shape == ()
        assert operation.hw.pp_delay == 1
        assert operation.internal_encode == 'private'
        assert len(operation.multipliers) == N - 1

        operation.reset()
        assert operation.timestep_cur == 0
        assert operation.regenerator.window_register.shape == (4,)
        assert operation.depth_register.shape == (4,)
        assert not bool(operation.regenerator._initialized)
        assert not bool(operation._depth_initialized)
        replay = torch.stack([operation(spike) for spike in stream])
        assert torch.equal(first_trace, replay), f'[{device}] reset replay changed'

        zeros = torch.zeros((8, 3, 2), dtype=global_config.stype, device=device)
        ones = torch.ones_like(zeros)
        zero_op = pow_regen(config).to(device)
        one_op = pow_regen(config).to(device)
        assert torch.count_nonzero(torch.stack([zero_op(x) for x in zeros])) == 0
        assert torch.equal(torch.stack([one_op(x) for x in ones]), ones)


def _comparison_report():
    """Print the full accuracy and factor-correlation design sweep."""
    values_cpu = torch.linspace(0.0, 1.0, 4096).reshape(64, 64)
    configs = [
        ('default', _operation_config(), (9, 0, 8)),
        ('4,4,1', _operation_config(lfsr_width=4, window=4, depth=1), (10, 0, 8)),
        ('4,4,2', _operation_config(lfsr_width=4, window=4, depth=2), (12, 0, 8)),
        ('8,8,3', _operation_config(lfsr_width=8, window=8, depth=3), (22, 0, 15)),
    ]

    for device in devices():
        values = values_cpu.to(device)
        for label, config, (ff_bits, rom_bits, lut6) in configs:
            for timestep in (64, 256, 1024):
                codec = encode({
                    'polarity': 'unipolar', 'timestep': timestep,
                    'generator': 'sobol', 'dim': 1,
                }).to(device)
                operations = {
                    n: pow_regen({**config, 'n': n}).to(device)
                    for n in range(2, 7)
                }
                counts = {n: torch.zeros_like(values) for n in operations}
                factor_source = encode_regen({
                    'polarity': 'unipolar',
                    'lfsr_width': config.get('lfsr_width', 3),
                    'window': config.get('window', 4),
                }).to(device)
                depth = config.get('depth', 1)
                history = []
                pair_metrics = [correlation().to(device) for _ in range(3)]

                for _ in range(timestep):
                    spike = codec(values)
                    for n, operation in operations.items():
                        counts[n].add_(operation(spike))
                    regenerated = factor_source(spike)
                    tap_1 = history[-depth] if len(history) >= depth else regenerated
                    tap_2 = history[-2 * depth] if len(history) >= 2 * depth else regenerated
                    pair_metrics[0](regenerated, tap_1)
                    pair_metrics[1](regenerated, tap_2)
                    pair_metrics[2](tap_1, tap_2)
                    history.append(regenerated.detach().clone())

                pairwise = [metric.correlation.mean().item() for metric in pair_metrics]
                print(
                    f'[{device}][{label}][T={timestep}] '
                    f'pairwise_scc={[round(value, 6) for value in pairwise]} '
                    f'ff_bits={ff_bits} rom_bits={rom_bits} lut6={lut6}'
                )
                for n, count in counts.items():
                    error = (count / timestep - values.pow(n)).abs()
                    maximum = error.max()
                    index = error.argmax()
                    print(
                        f'[{device}][{label}][n={n}][T={timestep}] '
                        f'mean_abs={error.mean().item():.6f} '
                        f'max_abs={maximum.item():.6f} '
                        f'max_x={values.flatten()[index].item():.6f}'
                    )

        caller_values = torch.linspace(0.05, 0.95, 19, device=device)
        for generator in ('sobol', 'lfsr_ext'):
            evidence = 'fidelity' if generator == 'sobol' else 'generator-sensitivity'
            codec = encode({
                'polarity': 'unipolar', 'timestep': 256,
                'generator': generator, 'dim': 1,
            }).to(device)
            operation = pow_regen(_operation_config()).to(device)
            count = torch.zeros_like(caller_values)
            for _ in range(256):
                count.add_(operation(codec(caller_values)))
            error = (count / 256 - caller_values.pow(N)).abs()
            print(
                f'[{device}][evidence={evidence}][caller_input={generator}]'
                f'[default][n={N}][T=256] '
                f'elements=19 mean_abs={error.mean().item():.6f} '
                f'max_abs={error.max().item():.6f}'
            )


def _extra_checks():
    _validation_checks()
    _structural_checks()
    _comparison_report()


CONFIG = {
    'polarities': ['unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': TIMESTEPS,
    'extra_checks': _extra_checks,
}


def test_pow_regen():
    """Verify unipolar power; bipolar multiplication requires XNOR semantics."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_pow_regen()

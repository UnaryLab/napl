import torch

from napl.sim.base import global_config
from napl.sim.metric import correlation
from napl.sim.operation import encode_regen, encode
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256
SEED = 7319


def make_operation(polarity, _timestep, _device):
    return encode_regen({'polarity': polarity})


def make_values(_polarity):
    generator = torch.Generator().manual_seed(SEED)
    return (torch.rand((16, 8), generator=generator),)


def make_random_perf_values(_polarity):
    generator = torch.Generator().manual_seed(SEED + 1)
    return (torch.rand((16384, 8), generator=generator),)


def analytic_reference(values, _polarity):
    return values[0]


def known_answer_case(_polarity):
    values = torch.tensor([[0.0, 1.0]])
    return (values,), values


def _bare_trace(stream, lfsr_width, window):
    """Return the registered-window comparator trace without encode_regen."""
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
    register = None
    outputs = []
    for timestep, input in enumerate(stream):
        if register is None:
            register = input.unsqueeze(0).repeat(window, *([1] * input.ndim))
        output = states[timestep % period].remainder(window).lt(
            register.sum(dim=0)
        ).type(global_config.stype)
        outputs.append(output)
        if timestep:
            register[timestep % window] = input
    return torch.stack(outputs)


def _validation_checks():
    """Reject bipolar use and every invalid LFSR/window sizing value exactly."""
    invalid = [
        ({'polarity': 'bipolar'},
         'Invalid polarity: <bipolar>; encode_regen supports unipolar only.'),
        ({'polarity': 'unipolar', 'lfsr_width': 2},
         'Invalid lfsr_width: <2>; legal values: an integer from 3 to 8.'),
        ({'polarity': 'unipolar', 'lfsr_width': 9},
         'Invalid lfsr_width: <9>; legal values: an integer from 3 to 8.'),
        ({'polarity': 'unipolar', 'lfsr_width': 4.0},
         'Invalid lfsr_width: <4.0>; legal values: an integer from 3 to 8.'),
        ({'polarity': 'unipolar', 'lfsr_width': True},
         'Invalid lfsr_width: <True>; legal values: an integer from 3 to 8.'),
        ({'polarity': 'unipolar', 'window': 1},
         'Invalid window: <1>; legal values: a power-of-two integer from 2 to 8.'),
        ({'polarity': 'unipolar', 'window': 3},
         'Invalid window: <3>; legal values: a power-of-two integer from 2 to 8.'),
        ({'polarity': 'unipolar', 'window': 16},
         'Invalid window: <16>; legal values: a power-of-two integer from 2 to 8.'),
        ({'polarity': 'unipolar', 'window': 4.0},
         'Invalid window: <4.0>; legal values: a power-of-two integer from 2 to 8.'),
        ({'polarity': 'unipolar', 'window': True},
         'Invalid window: <True>; legal values: a power-of-two integer from 2 to 8.'),
    ]
    for config, message in invalid:
        try:
            encode_regen(config)
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'encode_regen accepted invalid config {config}')


def _structural_checks():
    """Pin the comparator, registered window, state shape, and reset replay."""
    timesteps = 48
    generator = torch.Generator().manual_seed(SEED + 2)
    stream_cpu = torch.randint(
        0, 2, (timesteps, 3, 2), generator=generator, dtype=torch.int8,
    ).type(global_config.stype)

    for device in devices():
        stream = stream_cpu.to(device)
        operation = encode_regen({'polarity': 'unipolar'}).to(device)
        expected = _bare_trace(stream, 3, 4)
        first_trace = torch.stack([operation(spike) for spike in stream])

        # This exact trace is the removed-window-update and flipped-comparator bite.
        assert torch.equal(first_trace, expected), (
            f'[{device}] regenerator differs from the independent trace'
        )
        assert first_trace.shape == stream.shape
        assert operation.window_register.shape == (4, *stream.shape[1:])
        assert operation.lfsr_index.shape == ()
        histogram = torch.bincount(
            operation.lfsr_states.remainder(operation.window),
            minlength=operation.window,
        )
        assert torch.equal(
            histogram,
            torch.full_like(histogram, 2 ** operation.lfsr_width // operation.window),
        ), f'[{device}] extended cycle does not draw every threshold uniformly'
        assert operation.hw.pp_delay == 1
        assert operation.internal_encode == 'private'
        assert operation.correlation_o == {}

        operation.reset()
        assert operation.timestep_cur == 0
        assert operation.window_register.shape == (4,)
        assert operation.lfsr_index.item() == 0
        assert not bool(operation._initialized)
        replay = torch.stack([operation(spike) for spike in stream])
        assert torch.equal(first_trace, replay), f'[{device}] reset replay changed'

        zeros = torch.zeros((8, 3, 2), dtype=global_config.stype, device=device)
        ones = torch.ones_like(zeros)
        zero_op = encode_regen({'polarity': 'unipolar'}).to(device)
        one_op = encode_regen({'polarity': 'unipolar'}).to(device)
        assert torch.count_nonzero(torch.stack([zero_op(x) for x in zeros])) == 0
        assert torch.equal(torch.stack([one_op(x) for x in ones]), ones)


def _correlation_report():
    """Print input/output SCC; correlation metadata remains deliberately empty."""
    timestep = 1024
    values_cpu = torch.linspace(0.05, 0.95, 128).reshape(16, 8)
    for device in devices():
        codec = encode({
            'polarity': 'unipolar', 'timestep': timestep,
            'generator': 'sobol', 'dim': 1,
        }).to(device)
        operation = encode_regen({'polarity': 'unipolar'}).to(device)
        metric = correlation().to(device)
        values = values_cpu.to(device)
        for _ in range(timestep):
            input_spike = codec(values)
            metric(input_spike, operation(input_spike))
        print(
            f'[{device}][lfsr_width=3][window=4][T={timestep}] '
            f'mean_scc={metric.correlation.mean().item():.6f}'
        )


def _rate_evidence_report():
    """Print Sobol fidelity points and labeled input-generator mechanism evidence."""
    values_cpu = torch.linspace(0.0, 1.0, 4096).reshape(64, 64)
    configs = [
        ('3,4', {'lfsr_width': 3, 'window': 4}, (7, 0, 6)),
        ('4,4', {'lfsr_width': 4, 'window': 4}, (8, 0, 6)),
        ('8,8', {'lfsr_width': 8, 'window': 8}, (16, 0, 13)),
    ]
    for device in devices():
        values = values_cpu.to(device)
        for generator in ('sobol', 'lfsr', 'lfsr_ext'):
            evidence = 'fidelity' if generator == 'sobol' else 'mechanism'
            for label, sizing, (ff_bits, rom_bits, lut6) in configs:
                for timestep in (64, 256, 1024):
                    codec = encode({
                        'polarity': 'unipolar',
                        'timestep': timestep,
                        'generator': generator,
                        'dim': 1,
                    }).to(device)
                    operation = encode_regen({'polarity': 'unipolar', **sizing}).to(device)
                    count = torch.zeros_like(values)
                    for _ in range(timestep):
                        count.add_(operation(codec(values)))
                    signed_error = count / timestep - values
                    print(
                        f'[{device}][evidence={evidence}][input={generator}]'
                        f'[{label}][T={timestep}] '
                        f'mean_abs={signed_error.abs().mean().item():.6f} '
                        f'max_abs={signed_error.abs().max().item():.6f} '
                        f'mean_bias={signed_error.mean().item():.6f} '
                        f'ff_bits={ff_bits} rom_bits={rom_bits} lut6={lut6}'
                    )


def _extra_checks():
    _validation_checks()
    _structural_checks()
    _correlation_report()
    _rate_evidence_report()


CONFIG = {
    'polarities': ['unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_generators': lambda _polarity: ['sobol'],
    'timesteps': TIMESTEPS,
    'extra_checks': _extra_checks,
}


def test_encode_regen():
    """Verify unipolar regeneration; bipolar rates need a recentered estimator."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_encode_regen()

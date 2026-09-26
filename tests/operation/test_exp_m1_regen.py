import torch

from napl.sim.base import global_config
from napl.sim.operation import (
    delay, encode, exp_m1_regen, exp_n1, mul_gaines, mul_scale, encode_regen,
)
from napl.utils._shared_test import devices, streaming_suite


ORDER = 5
SEED = 7319


def _operation_config(**updates):
    config = {'polarity': 'unipolar', 'order': ORDER}
    config.update(updates)
    return config


def make_operation(polarity, _timestep, _device):
    return exp_m1_regen(_operation_config(polarity=polarity))


def make_values(_polarity):
    generator = torch.Generator().manual_seed(SEED)
    return (torch.rand((16, 8), generator=generator),)


def make_random_perf_values(_polarity):
    generator = torch.Generator().manual_seed(SEED + 1)
    return (torch.rand((16384, 8), generator=generator),)


def analytic_reference(values, _polarity):
    return torch.exp(values[0] - 1)


def known_answer_case(_polarity):
    values = torch.tensor([[0.0, 1.0]])
    return (values,), torch.exp(values - 1)


def _lfsr_ext_states(lfsr_width, device):
    maximal_period = 2 ** lfsr_width - 1
    sequence = encode({
        'polarity': 'unipolar', 'timestep': 2 ** lfsr_width,
        'generator': 'lfsr',
    })
    states = sequence.num_seq[:maximal_period]
    states = states.mul(2 ** lfsr_width).round().type(torch.long).to(device)
    insertion = int(torch.nonzero(states.eq(1), as_tuple=False)[0].item()) + 1
    return torch.cat((
        states[:insertion],
        torch.zeros(1, dtype=torch.long, device=device),
        states[insertion:],
    ))


def _bare_trace(stream, order, lfsr_width, window, depth):
    """Return the bit-exact nested-regen Horner topology without exp_m1_regen."""
    scalers = [
        mul_scale({
            'polarity': 'unipolar',
            'scale': 1 / k,
            'intwidth': 8,
            'fracwidth': 12,
        }).to(stream.device)
        for k in range(2, order + 1)
    ]
    n_prod = max(0, order - 1)
    multipliers = [
        mul_gaines({'polarity': 'unipolar'}).to(stream.device)
        for _ in range(n_prod)
    ]
    states = _lfsr_ext_states(lfsr_width, stream.device)
    period = 2 ** lfsr_width
    window_registers = [None] * n_prod
    delay_buffers = [None] * n_prod
    outputs = []

    for timestep, input in enumerate(stream):
        u = 1 - input.type(torch.int8)
        if order == 1:
            outputs.append((1 - u).type(global_config.stype))
            continue
        stage = 1 - scalers[-1](u)
        for k in range(order - 1, 0, -1):
            scaled = u if k == 1 else scalers[k - 2](u)
            nested = stage.type(global_config.stype)
            index = k - 1
            if window_registers[index] is None:
                window_registers[index] = nested.unsqueeze(0).repeat(
                    window, *([1] * nested.ndim)
                )
            regenerated = states[timestep % period].remainder(window).lt(
                window_registers[index].sum(dim=0)
            ).type(global_config.stype)
            if timestep:
                window_registers[index][timestep % window] = nested
            if delay_buffers[index] is None:
                delay_buffers[index] = [
                    torch.zeros_like(regenerated) for _ in range(depth)
                ]
            head = timestep % depth
            delayed = delay_buffers[index][head]
            delay_buffers[index][head] = regenerated.detach().clone()
            product = multipliers[index](scaled, delayed)
            stage = 1 - product
        outputs.append(stage.type(global_config.stype))

    return torch.stack(outputs)


def _validation_checks():
    """Reject every unsupported polarity and invalid sizing value exactly."""
    invalid = [
        ({'polarity': 'bipolar'},
         'Invalid polarity: <bipolar>; exp_m1_regen supports unipolar only.'),
        ({'order': 0},
         'Invalid order: <0>; legal values: an integer of at least 1.'),
        ({'order': -1},
         'Invalid order: <-1>; legal values: an integer of at least 1.'),
        ({'order': 1.5},
         'Invalid order: <1.5>; legal values: an integer of at least 1.'),
        ({'order': True},
         'Invalid order: <True>; legal values: an integer of at least 1.'),
        ({'order': '5'},
         'Invalid order: <5>; legal values: an integer of at least 1.'),
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
        ({'depth': 0},
         'Invalid depth: <0>; legal values: an integer of at least 1.'),
        ({'depth': -1},
         'Invalid depth: <-1>; legal values: an integer of at least 1.'),
        ({'depth': 1.0},
         'Invalid depth: <1.0>; legal values: an integer of at least 1.'),
        ({'depth': True},
         'Invalid depth: <True>; legal values: an integer of at least 1.'),
    ]
    for updates, message in invalid:
        try:
            exp_m1_regen(_operation_config(**updates))
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'exp_m1_regen accepted invalid config {updates}')


def _structural_checks():
    """Pin the regen+delay Horner wiring, rank-preserving state, and reset replay."""
    timesteps = 48
    generator = torch.Generator().manual_seed(SEED + 2)
    stream_cpu = torch.randint(
        0, 2, (timesteps, 3, 2), generator=generator, dtype=torch.int8,
    ).type(global_config.stype)
    config = _operation_config(lfsr_width=4, window=4, depth=2)

    for device in devices():
        stream = stream_cpu.to(device)
        operation = exp_m1_regen(config).to(device)
        expected = _bare_trace(stream, ORDER, 4, 4, 2)
        first_trace = torch.stack([operation(spike) for spike in stream])

        # These exact traces are the nested-operand and depth-delay bites.
        assert torch.equal(first_trace, expected), (
            f'[{device}] Horner regen topology differs from the independent trace'
        )
        assert first_trace.shape == stream.shape
        assert all(
            stage.accumulator.shape == stream.shape[1:]
            for stage in operation.scalers
        )
        assert all(
            stage.window_register.shape == (4, *stream.shape[1:])
            for stage in operation.regenerators
        )
        assert all(
            stage.lfsr_index.shape == ()
            for stage in operation.regenerators
        )
        assert all(
            stage.reg.shape == (2, *stream.shape[1:])
            for stage in operation.delays
        )
        assert operation.hw.pp_delay == 0
        assert operation.internal_encode == 'private'
        assert len(operation.scalers) == ORDER - 1
        assert len(operation.regenerators) == ORDER - 1
        assert len(operation.delays) == ORDER - 1
        assert len(operation.multipliers) == ORDER - 1

        operation.reset()
        assert operation.timestep_cur == 0
        assert all(stage.accumulator.shape == (1,) for stage in operation.scalers)
        assert all(
            stage.window_register.shape == (4,)
            and not bool(stage._initialized)
            for stage in operation.regenerators
        )
        assert all(stage.reg.shape == (2,) and stage.head.shape == ()
                   for stage in operation.delays)
        replay = torch.stack([operation(spike) for spike in stream])
        assert torch.equal(first_trace, replay), f'[{device}] reset replay changed'

        ones = torch.ones((8, 3, 2), dtype=global_config.stype, device=device)
        one_op = exp_m1_regen(config).to(device)
        assert torch.equal(torch.stack([one_op(x) for x in ones]), ones)


def _order_sensitive_checks():
    """Pin the highest Horner contribution with an order-5 versus order-4 trace."""
    # The default (3, 4, 1) regenerator absorbs the fifth-term rate gap, so this
    # check uses a wider window where the extra stage changes the emitted bits.
    timesteps = 256
    values_cpu = torch.linspace(0.0, 1.0, 16).reshape(4, 4)
    config = _operation_config(lfsr_width=5, window=8, depth=1)

    for device in devices():
        values = values_cpu.to(device)
        encoder = encode({
            'polarity': 'unipolar', 'timestep': timesteps,
            'generator': 'sobol', 'dim': 9,
        }).to(device)
        stream = torch.stack([encoder(values) for _ in range(timesteps)])
        higher = exp_m1_regen({**config, 'order': 5}).to(device)
        lower = exp_m1_regen({**config, 'order': 4}).to(device)
        higher_trace = torch.stack([higher(spike) for spike in stream])
        lower_trace = torch.stack([lower(spike) for spike in stream])
        different = torch.nonzero(
            higher_trace.ne(lower_trace).reshape(timesteps, -1).any(dim=1)
        ).flatten().cpu()
        expected_higher = _bare_trace(stream, 5, 5, 8, 1)
        expected_lower = _bare_trace(stream, 4, 5, 8, 1)
        assert torch.equal(higher_trace, expected_higher), (
            f'[{device}] order-5 class trace differs from the independent pipeline'
        )
        assert torch.equal(lower_trace, expected_lower), (
            f'[{device}] order-4 class trace differs from the independent pipeline'
        )
        expected_different = torch.nonzero(
            expected_higher.ne(expected_lower).reshape(timesteps, -1).any(dim=1)
        ).flatten().cpu()
        # The changed steps are derived from the independent order-5 and
        # order-4 bare traces, not a recorded snapshot.
        assert torch.equal(different, expected_different), (
            f'[{device}] class changed-step set differs from the bare-trace oracle'
        )
        assert different.numel() > 0, f'[{device}] order-5 contribution is absent'


def _taylor(values, order):
    u = 1 - values
    result = torch.ones_like(values)
    power = torch.ones_like(values)
    for degree in range(1, order + 1):
        power = power * (-u) / degree
        result = result + power
    return result


def _horner_regen_rate(values, encoder, timestep, regen_nested, order=ORDER):
    scalers = [
        mul_scale({
            'polarity': 'unipolar',
            'scale': 1 / k,
            'intwidth': 8,
            'fracwidth': 12,
        }).to(values.device)
        for k in range(2, order + 1)
    ]
    n_prod = max(0, order - 1)
    regenerators = [
        encode_regen({
            'polarity': 'unipolar', 'lfsr_width': 3, 'window': 4,
        }).to(values.device)
        for _ in range(n_prod)
    ]
    delays = [delay({'depth': 1}).to(values.device) for _ in range(n_prod)]
    multipliers = [
        mul_gaines({'polarity': 'unipolar'}).to(values.device)
        for _ in range(n_prod)
    ]
    count = torch.zeros_like(values)
    for _ in range(timestep):
        spike = encoder(values)
        u = 1 - spike.type(torch.int8)
        if order == 1:
            count = count + (1 - u).float()
            continue
        stage = 1 - scalers[-1](u)
        for k in range(order - 1, 0, -1):
            scaled = u if k == 1 else scalers[k - 2](u)
            index = k - 1
            if regen_nested:
                regenerated = regenerators[index](stage)
                delayed = delays[index](regenerated)
                product = multipliers[index](scaled, delayed)
            else:
                regenerated = regenerators[index](scaled)
                delayed = delays[index](regenerated)
                product = multipliers[index](delayed, stage)
            stage = 1 - product
        count = count + stage.float()
    return count / timestep


def _comparison_report():
    """Print operand-choice, order-error, and Sobol caller evidence."""
    values_cpu = torch.linspace(0.0, 1.0, 128).reshape(16, 8)
    exact_cpu = torch.exp(values_cpu - 1)
    timestep = 256

    for device in devices():
        values = values_cpu.to(device)
        encoder = encode({
            'polarity': 'unipolar', 'timestep': timestep,
            'generator': 'sobol', 'dim': 9,
        }).to(device)
        stream = [encoder(values) for _ in range(timestep)]

        for label, regen_nested in (
            ('A regen nested', True),
            ('B regen scaled', False),
        ):
            encoder.reset()
            rate = _horner_regen_rate(values, encoder, timestep, regen_nested)
            err = rate.cpu() - exact_cpu
            print(
                f'[{device}][operand] {label}: '
                f'rmse={err.pow(2).mean().sqrt().item():.8f} '
                f'max_abs={err.abs().max().item():.8f}'
            )
        print(f'[{device}][operand] selected=A')

        family = exp_n1({
            'polarity': 'unipolar', 'timestep': timestep,
            'generator': 'sobol', 'dim': 17,
        }).to(device)
        family_rate = torch.stack(
            [family(1 - spike) for spike in stream]
        ).type(global_config.ntype).mean(0)
        family_exact_rmse = (
            family_rate.cpu() - exact_cpu
        ).pow(2).mean().sqrt().item()

        for order in (1, 3, 5, 7):
            operation = exp_m1_regen(_operation_config(order=order)).to(device)
            rate = torch.stack(
                [operation(spike) for spike in stream]
            ).type(global_config.ntype).mean(0).cpu()
            polynomial = _taylor(values_cpu, order)
            truncation_rmse = (polynomial - exact_cpu).pow(2).mean().sqrt().item()
            sampling_rmse = (rate - polynomial).pow(2).mean().sqrt().item()
            exact_rmse = (rate - exact_cpu).pow(2).mean().sqrt().item()
            family_rmse = (rate - family_rate.cpu()).pow(2).mean().sqrt().item()
            print(
                f'[{device}] order={order}, N={timestep}, '
                f'truncation_rmse={truncation_rmse:.6f}, '
                f'sampling_rmse={sampling_rmse:.6f}, '
                f'exact_rmse={exact_rmse:.6f}, '
                f'exp_n1_complement_rmse={family_rmse:.6f}, '
                f'exp_n1_exact_rmse={family_exact_rmse:.6f}'
            )

        caller_values = torch.linspace(0.05, 0.95, 19, device=device)
        for generator in ('sobol', 'lfsr_ext'):
            evidence = 'fidelity' if generator == 'sobol' else 'generator-sensitivity'
            codec = encode({
                'polarity': 'unipolar', 'timestep': timestep,
                'generator': generator, 'dim': 9,
            }).to(device)
            operation = exp_m1_regen(_operation_config()).to(device)
            count = torch.zeros_like(caller_values)
            for _ in range(timestep):
                count.add_(operation(codec(caller_values)))
            error = (count / timestep - torch.exp(caller_values - 1)).abs()
            print(
                f'[{device}][evidence={evidence}][caller_input={generator}]'
                f'[default][order={ORDER}][T={timestep}] '
                f'elements=19 mean_abs={error.mean().item():.6f} '
                f'max_abs={error.max().item():.6f}'
            )


def _extra_checks():
    _validation_checks()
    _structural_checks()
    _order_sensitive_checks()
    _comparison_report()


CONFIG = {
    'polarities': ['unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [9],
    'timesteps': 256,
    'extra_checks': _extra_checks,
}


def test_exp_m1_regen():
    """Verify the unipolar regenerated Horner exp(x-1) approximation."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_exp_m1_regen()

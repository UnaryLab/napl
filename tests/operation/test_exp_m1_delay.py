import torch

from napl.sim.base import global_config
from napl.sim.metric import correlation
from napl.sim.operation import encode, exp_m1_delay, exp_n1, mul_gaines, mul_scale
from napl.utils._shared_test import devices, streaming_suite


ORDER = 5
SEED = 7319


def _operation_config(order, depth=1):
    return {
        'polarity': 'unipolar',
        'order': order,
        'depth': depth,
    }


def make_operation(polarity, _timestep, _device):
    config = _operation_config(ORDER)
    config['polarity'] = polarity
    return exp_m1_delay(config)


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


def _bare_horner(order, depth):
    scalers = [
        mul_scale({
            'polarity': 'unipolar',
            'scale': 1 / k,
            'intwidth': 8,
            'fracwidth': 12,
        })
        for k in range(2, order + 1)
    ]
    multipliers = [
        mul_gaines({'polarity': 'unipolar'}) for _ in range(max(0, order - 1))
    ]
    histories = None

    def forward(input):
        nonlocal histories
        u = 1 - input.type(torch.int8)
        if order == 1:
            return (1 - u).type(global_config.stype)
        if histories is None:
            histories = [
                [torch.zeros_like(input) for _ in range(depth)]
                for _ in range(order - 1)
            ]
        stage = 1 - scalers[-1](u)
        for k in range(order - 1, 0, -1):
            scaled = u if k == 1 else scalers[k - 2](u)
            nested = histories[k - 1][-1]
            histories[k - 1] = [stage.detach().clone(), *histories[k - 1][:-1]]
            product = multipliers[k - 1](scaled, nested)
            stage = 1 - product
        return stage.type(global_config.stype)

    return scalers, multipliers, forward


def _validation_checks():
    """Reject bipolar polarity, illegal order and depth, and missing keys."""
    try:
        exp_m1_delay({**_operation_config(ORDER), 'polarity': 'bipolar'})
    except AssertionError as error:
        assert str(error) == (
            'Invalid polarity: <bipolar>; exp_m1_delay supports unipolar only.'
        )
    else:
        raise AssertionError('exp_m1_delay must reject bipolar input')

    for order in (0, -1, 1.5, True, '5'):
        message = f'Invalid order: <{order}>; legal values: an integer of at least 1.'
        try:
            exp_m1_delay({**_operation_config(ORDER), 'order': order})
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'exp_m1_delay must reject order={order!r}')

    for depth in (0, -1, 1.5, True):
        message = f'Invalid depth: <{depth}>; legal values: an integer of at least 1.'
        try:
            exp_m1_delay({**_operation_config(ORDER), 'depth': depth})
        except AssertionError as error:
            assert str(error) == message, f'{str(error)!r} != {message!r}'
            continue
        raise AssertionError(f'exp_m1_delay must reject depth={depth!r}')

    required = {'polarity': 'unipolar', 'order': 5}
    for missing in ('polarity', 'order'):
        config = dict(required)
        del config[missing]
        try:
            exp_m1_delay(config)
        except AssertionError as error:
            assert str(error) == f'Missing key <{missing}> in the input configuration.'
            continue
        raise AssertionError(f'exp_m1_delay must require {missing}')


def _structural_checks():
    """Pin the delayed Horner wiring, rank-preserving state, metadata, and reset replay."""
    depth = 3
    timesteps = 24
    generator = torch.Generator().manual_seed(SEED + 2)
    stream_cpu = torch.randint(
        0, 2, (timesteps, 3, 2, 4), generator=generator, dtype=torch.int8,
    ).type(global_config.stype)

    for device in devices():
        operation = exp_m1_delay(_operation_config(ORDER, depth)).to(device)
        scalers, multipliers, reference = _bare_horner(ORDER, depth)
        for module in (*scalers, *multipliers):
            module.to(device)
        stream = stream_cpu.to(device)

        first_trace = []
        for spike in stream:
            output = operation(spike)
            expected = reference(spike)
            assert torch.equal(output, expected), (
                f'[{device}] delayed Horner differs from the bare pipeline'
            )
            assert output.shape == spike.shape
            first_trace.append(output.detach().clone())

        assert operation.hw.pp_delay == 0
        assert operation.internal_encode == 'none'
        assert operation.correlation_i == {}
        assert operation.correlation_o == {}
        assert len(operation.scalers) == max(0, ORDER - 1)
        assert len(operation.delays) == ORDER - 1
        assert len(operation.multipliers) == ORDER - 1
        assert all(
            stage.reg.shape == (depth, *stream.shape[1:])
            for stage in operation.delays
        )
        assert all(
            stage.accumulator.shape == stream.shape[1:]
            for stage in operation.scalers
        )

        operation.reset()
        assert operation.timestep_cur == 0
        assert all(
            stage.reg.shape == (depth,) and stage.head.shape == ()
            for stage in operation.delays
        )
        assert all(stage.accumulator.shape == (1,) for stage in operation.scalers)
        replay = [operation(spike) for spike in stream]
        assert all(torch.equal(first, second) for first, second in zip(first_trace, replay))


def _combinational_path_checks():
    """Pin the current-input combinational path behind pp_delay zero."""
    depth = 3
    ones = torch.ones((2, 3), dtype=global_config.stype)
    zeros = torch.zeros_like(ones)

    for device in devices():
        low = exp_m1_delay(_operation_config(1, depth)).to(device)
        high = exp_m1_delay(_operation_config(1, depth)).to(device)
        fill = ones.to(device)
        for _ in range(depth):
            low(fill)
            high(fill)
        low_output = low(zeros.to(device))
        high_output = high(fill)
        assert not torch.equal(low_output, high_output), (
            f'[{device}] output ignored the current input after the delay filled'
        )
        assert low.hw.pp_delay == 0


def _order_sensitive_checks():
    """Pin the highest Horner contribution with an order-5 versus order-4 trace."""
    timesteps = 64
    stream = torch.zeros(
        (timesteps, 3, 4), dtype=global_config.stype,
    )

    for device in devices():
        higher = exp_m1_delay(_operation_config(5)).to(device)
        lower = exp_m1_delay(_operation_config(4)).to(device)
        higher_trace = torch.stack([higher(spike.to(device)) for spike in stream])
        lower_trace = torch.stack([lower(spike.to(device)) for spike in stream])
        assert not torch.equal(higher_trace, lower_trace), (
            f'[{device}] order-5 contribution is absent'
        )


def _taylor(values, order):
    u = 1 - values
    result = torch.ones_like(values)
    power = torch.ones_like(values)
    for degree in range(1, order + 1):
        power = power * (-u) / degree
        result = result + power
    return result


def _rate_and_scc(operation, stream, device):
    metrics = [correlation().to(device) for _ in operation.multipliers]
    handles = [
        stage.register_forward_pre_hook(
            lambda _module, inputs, index=index: metrics[index](inputs[0], inputs[1])
        )
        for index, stage in enumerate(operation.multipliers)
    ]
    rate = torch.stack(
        [operation(spike) for spike in stream]
    ).type(global_config.ntype).mean(0)
    for handle in handles:
        handle.remove()
    if metrics:
        mean_abs_scc = torch.stack([
            metric.correlation.abs().mean() for metric in metrics
        ]).mean().item()
    else:
        mean_abs_scc = float('nan')
    return rate, mean_abs_scc


def _order_error_report():
    """Print order error versus closed-form exp(x-1) and the Taylor polynomial."""
    values_cpu = torch.linspace(0.0, 1.0, 128).reshape(16, 8)
    exact_cpu = torch.exp(values_cpu - 1)

    for device in devices():
        values = values_cpu.to(device)
        for timesteps in (256, 1024, 4096):
            encoder = encode({
                'polarity': 'unipolar', 'timestep': timesteps,
                'generator': 'sobol', 'dim': 9,
            }).to(device)
            stream = [encoder(values) for _ in range(timesteps)]

            family = exp_n1({
                'polarity': 'unipolar', 'timestep': timesteps,
                'generator': 'sobol', 'dim': 17,
            }).to(device)
            family_rate = torch.stack(
                [family(1 - spike) for spike in stream]
            ).type(global_config.ntype).mean(0)
            family_exact_rmse = (
                family_rate.cpu() - exact_cpu
            ).pow(2).mean().sqrt().item()

            for order in (1, 3, 5, 7):
                operation = exp_m1_delay(_operation_config(order)).to(device)
                rate, mean_abs_scc = _rate_and_scc(operation, stream, device)
                rate = rate.cpu()
                polynomial = _taylor(values_cpu, order)
                truncation_rmse = (
                    polynomial - exact_cpu
                ).pow(2).mean().sqrt().item()
                sampling_rmse = (
                    rate - polynomial
                ).pow(2).mean().sqrt().item()
                exact_rmse = (rate - exact_cpu).pow(2).mean().sqrt().item()
                family_rmse = (
                    rate - family_rate.cpu()
                ).pow(2).mean().sqrt().item()
                print(
                    f'[{device}] order={order}, N={timesteps}, depth=1, '
                    f'truncation_rmse={truncation_rmse:.6f}, '
                    f'sampling_rmse={sampling_rmse:.6f}, '
                    f'exact_rmse={exact_rmse:.6f}, '
                    f'exp_n1_complement_rmse={family_rmse:.6f}, '
                    f'exp_n1_exact_rmse={family_exact_rmse:.6f}, '
                    f'mean_abs_scaled_nested_scc={mean_abs_scc:.6f}'
                )


def _depth_sweep_report():
    """Print depth-sweep error versus closed-form exp(x-1) and the Taylor polynomial."""
    values_cpu = torch.linspace(0.0, 1.0, 128).reshape(16, 8)
    exact_cpu = torch.exp(values_cpu - 1)
    polynomial = _taylor(values_cpu, ORDER)
    truncation_rmse = (polynomial - exact_cpu).pow(2).mean().sqrt().item()

    for device in devices():
        values = values_cpu.to(device)
        for timesteps in (256, 1024):
            encoder = encode({
                'polarity': 'unipolar', 'timestep': timesteps,
                'generator': 'sobol', 'dim': 9,
            }).to(device)
            stream = [encoder(values) for _ in range(timesteps)]
            for depth in (1, 2, 3, 5, 8):
                operation = exp_m1_delay(_operation_config(ORDER, depth)).to(device)
                rate, mean_abs_scc = _rate_and_scc(operation, stream, device)
                rate = rate.cpu()
                sampling_rmse = (rate - polynomial).pow(2).mean().sqrt().item()
                exact_rmse = (rate - exact_cpu).pow(2).mean().sqrt().item()
                print(
                    f'[{device}] order={ORDER}, N={timesteps}, depth={depth}, '
                    f'truncation_rmse={truncation_rmse:.6f}, '
                    f'sampling_rmse={sampling_rmse:.6f}, '
                    f'exact_rmse={exact_rmse:.6f}, '
                    f'mean_abs_scaled_nested_scc={mean_abs_scc:.6f}'
                )


def _extra_checks():
    _validation_checks()
    _structural_checks()
    _combinational_path_checks()
    _order_sensitive_checks()
    _order_error_report()
    _depth_sweep_report()


# The independent delayed Horner oracle pins the per-timestep nested product,
# catching a bypassed nested path, a skipped delay update, or a product whose
# emitted output is bypassed.
CONFIG = {
    'polarities': ['unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [9],
    'timesteps': 256,
    'state_timesteps': 16,
    'warmup_runs': 1,
    'trials': 3,
    'extra_checks': _extra_checks,
}


def test_exp_m1_delay():
    """Verify the delayed unipolar exp(x-1) Horner chain over its full input range."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_exp_m1_delay()

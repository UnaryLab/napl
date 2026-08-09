import math

import torch

from napl.sim.base import global_config
from napl.utils._shared_test import devices, streaming_suite
from napl.sim.operation import add_any, add_any_dyn


ENTRY = 8
SCALE = ENTRY
SCALE_MAX = 16
WIDTH = 20
SEED = 1234


def _spike_stream(timesteps, shape, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 2, (timesteps,) + shape, generator=generator, dtype=torch.int64
    ).type(global_config.stype)


def make_operation(polarity, _timestep, _device):
    return add_any_dyn({'polarity': polarity, 'scale_max': SCALE_MAX, 'width': WIDTH})


def make_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(lo, hi, 512).reshape(64, ENTRY),)


def make_performance_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(lo, hi, 16384 * ENTRY).reshape(16384, ENTRY),)


def analytic_reference(values, _polarity):
    return values[0].mean(dim=-1)


def known_answer_case(_polarity):
    values = torch.ones((ENTRY, ENTRY))
    return (values,), values.mean(dim=-1), 2.0 / math.sqrt(256)


CONFIG = {
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'polarities': ['unipolar', 'bipolar'],
    'timesteps': 256,
    'tolerance_scale': 2.0,
    'apply_operation': lambda operation, spikes: operation(
        spikes[0], SCALE, entry=ENTRY, dim=-1
    ),
}


def test_add_any_dyn():
    """Verify add_any_dyn at a held scale against analytic and known-answer streams."""
    streaming_suite(CONFIG)


def test_add_any_dyn_matches_add_any_at_constant_scale():
    """Verify add_any_dyn reproduces add_any bit-exactly when the scale never changes."""
    timesteps = 128
    stream_cpu = _spike_stream(timesteps, (16, ENTRY))

    compared = 0
    for device in devices():
        stream = stream_cpu.to(device)
        for polarity in ('unipolar', 'bipolar'):
            # scale below, at, and above the adder fan-in.
            for scale in (4, ENTRY, 16):
                reference = add_any(
                    {'polarity': polarity, 'scale': scale, 'width': WIDTH}
                ).to(device)
                candidate = add_any_dyn(
                    {'polarity': polarity, 'scale_max': SCALE_MAX, 'width': WIDTH}
                ).to(device)
                for timestep in range(timesteps):
                    expected = reference(stream[timestep], dim=-1)
                    observed = candidate(stream[timestep], scale, dim=-1)
                    assert observed.shape == (16,), (
                        f'[{device}] {polarity} scale={scale} output shape '
                        f'{tuple(observed.shape)} != (16,)'
                    )
                    assert torch.equal(observed, expected), (
                        f'[{device}] {polarity} scale={scale} diverged at '
                        f'timestep {timestep}'
                    )
                assert candidate.accumulator.shape == (16,)
                assert torch.equal(candidate.accumulator, reference.accumulator)
                compared += 1
        print(f'[{device}] add_any_dyn matched add_any on 6 polarity/scale pairs')

    assert compared == 6 * len(devices())


def test_add_any_dyn_conserves_mass_across_a_scale_change():
    """Verify fired mass plus the accumulator equals the input count across a scale change."""
    timesteps = 64
    early_scale, late_scale = ENTRY, 3
    stream_cpu = _spike_stream(timesteps, (4, ENTRY), seed=SEED + 1)

    for device in devices():
        stream = stream_cpu.to(device)
        operation = add_any_dyn(
            {'polarity': 'unipolar', 'scale_max': SCALE_MAX, 'width': WIDTH}
        ).to(device)

        fired_mass = torch.zeros(4, dtype=global_config.ntype, device=device)
        for timestep in range(timesteps):
            scale = early_scale if timestep < timesteps // 2 else late_scale
            output = operation(stream[timestep], scale, dim=-1)
            fired_mass.add_(output.type(global_config.ntype), alpha=scale)

        total_input = stream.type(global_config.ntype).sum(dim=(0, -1))
        residual = total_input - fired_mass
        # The accumulator drains by at most the pre-change scale, so it ends the
        # first half below ENTRY and then gains at most (ENTRY - late_scale) per
        # remaining timestep.
        bound = (ENTRY - 1) + (timesteps // 2) * (ENTRY - late_scale)
        max_residual = residual.max().item()

        assert torch.equal(residual, operation.accumulator), (
            f'[{device}] residual {residual.tolist()} != accumulator '
            f'{operation.accumulator.tolist()}'
        )
        assert residual.min().item() >= 0, f'[{device}] negative residual'
        assert max_residual <= bound, (
            f'[{device}] residual {max_residual} exceeds bound {bound}'
        )
        assert max_residual < operation.acc_max, f'[{device}] accumulator clamped'
        print(
            f'[{device}] input={total_input.sum().item():.0f}, '
            f'fired={fired_mass.sum().item():.0f}, '
            f'residual max={max_residual:.0f}, bound={bound}'
        )


def test_add_any_dyn_conserves_mass_across_a_scale_change_bipolar():
    """Verify the bipolar offset tracks the current call scale across a scale change."""
    timesteps = 64
    early_scale, late_scale = ENTRY, 3
    stream_cpu = _spike_stream(timesteps, (4, ENTRY), seed=SEED + 1)

    for device in devices():
        stream = stream_cpu.to(device)
        operation = add_any_dyn(
            {'polarity': 'bipolar', 'scale_max': SCALE_MAX, 'width': WIDTH}
        ).to(device)

        fired_mass = torch.zeros(4, dtype=global_config.ntype, device=device)
        expected_total = torch.zeros(4, dtype=global_config.ntype, device=device)
        for timestep in range(timesteps):
            scale = early_scale if timestep < timesteps // 2 else late_scale
            expected_total.add_(
                stream[timestep].type(global_config.ntype).sum(dim=-1)
                - (ENTRY - scale) / 2
            )
            output = operation(stream[timestep], scale, dim=-1)
            fired_mass.add_(output.type(global_config.ntype), alpha=scale)

        residual = expected_total - fired_mass
        assert torch.equal(residual, operation.accumulator), (
            f'[{device}] residual {residual.tolist()} != accumulator '
            f'{operation.accumulator.tolist()}'
        )
        assert residual.abs().max().item() < operation.acc_max, (
            f'[{device}] accumulator clamped'
        )
        print(
            f'[{device}] bipolar expected={expected_total.sum().item():.1f}, '
            f'fired={fired_mass.sum().item():.0f}, '
            f'residual max={residual.max().item():.1f}'
        )


def test_add_any_dyn_saturates_at_the_accumulator_rail():
    """Verify a narrow accumulator clamps at acc_max before the carry is subtracted."""
    timesteps = 40
    width = 6
    scale = 1
    input_cpu = torch.ones((4, ENTRY), dtype=global_config.stype)

    for device in devices():
        input = input_cpu.to(device)
        operation = add_any_dyn(
            {'polarity': 'unipolar', 'scale_max': SCALE_MAX, 'width': width}
        ).to(device)

        for _ in range(timesteps):
            output = operation(input, scale, dim=-1)

        # Inflow of ENTRY per timestep outruns the drain of scale, so the clamp
        # holds the accumulator at acc_max and the carry leaves acc_max - scale.
        railed = operation.acc_max - scale
        assert torch.equal(
            operation.accumulator,
            torch.full_like(operation.accumulator, railed),
        ), (
            f'[{device}] accumulator {operation.accumulator.tolist()} != '
            f'acc_max - scale {railed}'
        )
        assert torch.equal(output, torch.ones_like(output)), (
            f'[{device}] output {output.tolist()} not all ones at saturation'
        )
        print(
            f'[{device}] saturated: acc_max={operation.acc_max}, '
            f'accumulator={operation.accumulator[0].item():.0f}'
        )


def test_add_any_dyn_rejects_unreachable_scale_max():
    """Reject a maximum carry threshold above the signed accumulator rail."""
    try:
        add_any_dyn({'polarity': 'unipolar', 'scale_max': 128, 'width': 8})
    except AssertionError:
        return
    raise AssertionError('add_any_dyn must reject a scale_max above acc_max')


def test_add_any_dyn_rejects_non_int_config():
    """Reject a non-int accumulator width or scale_max."""
    for key, value in (('width', 10.5), ('scale_max', 2.0)):
        config = {'polarity': 'unipolar', 'scale_max': SCALE_MAX, 'width': WIDTH}
        config[key] = value
        try:
            add_any_dyn(config)
        except AssertionError:
            continue
        raise AssertionError(f'add_any_dyn must reject {key}={value}')


def test_add_any_dyn_rejects_out_of_range_call_scale():
    """Reject a per-call scale outside one to scale_max."""
    input = torch.ones((2, ENTRY), dtype=global_config.stype)
    operation = add_any_dyn(
        {'polarity': 'unipolar', 'scale_max': SCALE_MAX, 'width': WIDTH}
    )

    # A legal call first, so the rejections below land on non-first calls.
    operation(input, 4, dim=-1)

    for scale in (0, -1, SCALE_MAX + 1, True, False):
        try:
            operation(input, scale, dim=-1)
        except AssertionError:
            continue
        raise AssertionError(f'add_any_dyn must reject a per-call scale of {scale}')


if __name__ == '__main__':
    test_add_any_dyn()
    test_add_any_dyn_matches_add_any_at_constant_scale()
    test_add_any_dyn_conserves_mass_across_a_scale_change()
    test_add_any_dyn_conserves_mass_across_a_scale_change_bipolar()
    test_add_any_dyn_saturates_at_the_accumulator_rail()
    test_add_any_dyn_rejects_unreachable_scale_max()
    test_add_any_dyn_rejects_non_int_config()
    test_add_any_dyn_rejects_out_of_range_call_scale()

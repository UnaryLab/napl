import math

import torch
import torch.nn.functional as F

from napl.sim.base import global_config
from napl.sim.module import avgpool2d_ugemm
from napl.sim.operation import decode, encode
from napl.utils._shared_test import clone_inputs, devices, streaming_suite


_KERNEL_SIZE = 2
_SHAPE = (4, 3, 8, 8)
_STRIDE = 1
_TIMESTEPS = 256


def make_operation(polarity, timestep, device):
    return avgpool2d_ugemm(
        _KERNEL_SIZE,
        config={'polarity': polarity},
    ).to(device)


def make_values(polarity):
    low = 0.0 if polarity == 'unipolar' else -1.0
    return (
        torch.linspace(
            low,
            1.0,
            math.prod(_SHAPE),
            dtype=global_config.ntype,
        ).reshape(_SHAPE),
    )


def make_performance_values(polarity):
    values = make_values(polarity)[0]
    return (values.repeat(256, 1, 1, 1),)


def analytic_reference(values, polarity):
    return F.avg_pool2d(values[0], _KERNEL_SIZE)


def known_answer_case(polarity):
    value = torch.full((1, 1, 4, 4), 0.5, dtype=global_config.ntype)
    return (value,), torch.full((1, 1, 2, 2), 0.5), 0.0


def check_stride():
    """Pool with an explicit stride and compare with strided F.avg_pool2d."""
    torch.manual_seed(0)
    tolerance = 4.0 / math.sqrt(_TIMESTEPS)
    for polarity in ['unipolar', 'bipolar']:
        low = 0.0 if polarity == 'unipolar' else -1.0
        values_cpu = torch.linspace(
            low, 1.0, math.prod(_SHAPE), dtype=global_config.ntype,
        ).reshape(_SHAPE)
        reference = F.avg_pool2d(values_cpu, _KERNEL_SIZE, stride=_STRIDE)
        for device in devices():
            values, = clone_inputs((values_cpu,), device)
            codec = {'polarity': polarity, 'timestep': _TIMESTEPS,
                     'generator': 'sobol', 'dim': 1}
            enc = encode(codec).to(device)
            pool = avgpool2d_ugemm(
                _KERNEL_SIZE, stride=_STRIDE, config={'polarity': polarity},
            ).to(device)
            dec = decode(codec).to(device)
            for _ in range(_TIMESTEPS):
                dec(pool(enc(values)))
            result = dec.spike_value.detach().cpu()
            assert result.shape == reference.shape
            rmse = (result - reference).pow(2).mean().sqrt().item()
            assert rmse <= tolerance, (
                f'[{device}][{polarity}] stride={_STRIDE} rmse={rmse:.6f}, '
                f'bound={tolerance:.6f}'
            )
            print(
                f'[{device}][{polarity}] stride={_STRIDE}, seed=0, '
                f'N={_TIMESTEPS}, rmse={rmse:.6f}, bound={tolerance:.6f}'
            )


def check_min_width():
    """Pool a non-power-of-two window at the smallest legal accumulator width."""
    kernel_size = 3
    kernel_area = kernel_size * kernel_size
    width = math.ceil(math.log2(2 * kernel_area)) + 1
    tolerance = 4.0 / _TIMESTEPS
    try:
        avgpool2d_ugemm(kernel_size, config={'polarity': 'unipolar', 'width': width - 1})
    except AssertionError:
        pass
    else:
        raise AssertionError(f'width <{width - 1}> must be rejected for kernel area <{kernel_area}>')
    values_cpu = torch.linspace(
        0.0, 1.0, math.prod(_SHAPE), dtype=global_config.ntype,
    ).reshape(_SHAPE)
    reference = F.avg_pool2d(values_cpu, kernel_size)
    for device in devices():
        values, = clone_inputs((values_cpu,), device)
        codec = {'polarity': 'unipolar', 'timestep': _TIMESTEPS,
                 'generator': 'sobol', 'dim': 1}
        enc = encode(codec).to(device)
        pool = avgpool2d_ugemm(
            kernel_size, config={'polarity': 'unipolar', 'width': width},
        ).to(device)
        dec = decode(codec).to(device)
        for _ in range(_TIMESTEPS):
            dec(pool(enc(values)))
        result = dec.spike_value.detach().cpu()
        assert result.shape == reference.shape
        rmse = (result - reference).pow(2).mean().sqrt().item()
        assert rmse <= tolerance, (
            f'[{device}][unipolar] kernel={kernel_size} width={width} '
            f'rmse={rmse:.6f}, bound={tolerance:.6f}'
        )
        print(
            f'[{device}][unipolar] kernel={kernel_size}, width={width}, '
            f'N={_TIMESTEPS}, rmse={rmse:.6f}, bound={tolerance:.6f}'
        )


def extra_checks():
    """Run the stride and minimum-width checks."""
    check_stride()
    check_min_width()


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 4.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'extra_checks': extra_checks,
}


def test_avgpool2d_ugemm():
    """Verify avgpool2d_ugemm against analytic and known-answer streams, including stride, reset, and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_avgpool2d_ugemm()
    print('Test passed.')

"""Test the hybrid unary-binary convolution.

Gradients are exempt: the layer wraps ``conv_ugemm``, whose spike comparison is
not differentiable and carries no straight-through estimator, so no gradient ever
reaches its weight and bias parameters and gate 6 has no defining equation to
check.
"""

import torch
import torch.nn.functional as F

from napl.sim.base import global_config
from napl.sim.module import conv_ugemm, conv_ugemm_hub
from napl.sim.operation import decode, encode
from napl.utils._shared_test import benchmark, devices


TIMESTEP = 512
BATCH, IN_CHANNELS, OUT_CHANNELS, SIDE, KERNEL = 2, 2, 3, 6, 3
PADDING = 1
ENTRY = IN_CHANNELS * KERNEL * KERNEL + 1
# Per-polarity fidelity bounds at about 1.5x the observed rmse (0.000694 unipolar,
# 0.002345 bipolar). The Sobol streams make the observed error deterministic and
# identical to within 4e-9 across devices for the fixed inputs, weights, and dims below, and the
# two polarities sit about 3x apart, so each polarity carries its own bound. The
# repo-standard 3/sqrt(N) = 0.132583 is the bound for a randomly rate-coded stream;
# Sobol streams converge nearer 1/N than 1/sqrt(N), so that bound sits about 65x
# above what this wrapper reaches and would pass a small wiring, scale, or sign
# error. These bounds trip on a 1% weight-scale error in the unipolar case and a
# 3% one in the bipolar case.
RMSE_BOUND = {'unipolar': 0.001, 'bipolar': 0.0035}
POLARITIES = ['unipolar', 'bipolar']


def _codec_config(polarity, timestep=TIMESTEP, dim=2):
    return {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': dim}


def _core_config():
    return {'scale': None, 'width': 12}


def _parameters(polarity):
    count = OUT_CHANNELS * IN_CHANNELS * KERNEL * KERNEL
    weight = torch.linspace(-0.5, 0.5, count, dtype=global_config.ntype).view(
        OUT_CHANNELS, IN_CHANNELS, KERNEL, KERNEL)
    bias = torch.linspace(0.4, -0.4, OUT_CHANNELS, dtype=global_config.ntype)
    if polarity == 'unipolar':
        weight = weight.abs()
        bias = bias.abs()
    return weight, bias


def _input_value(polarity):
    low = 0.0 if polarity == 'unipolar' else -1.0
    count = BATCH * IN_CHANNELS * SIDE * SIDE
    return torch.linspace(low, 1.0, count, dtype=global_config.ntype).view(
        BATCH, IN_CHANNELS, SIDE, SIDE)


def _make_hub(polarity, device, timestep=TIMESTEP, dim=2):
    weight, bias = _parameters(polarity)
    return conv_ugemm_hub(weight.clone(), bias.clone(), stride=1, padding=PADDING, dilation=1,
                          codec_config=_codec_config(polarity, timestep, dim),
                          core_config=_core_config()).to(device)


def _make_bare(polarity, device, timestep=TIMESTEP, dim=2):
    """Return the manual encode, bare core, and decode composition the layer wraps."""
    weight, bias = _parameters(polarity)
    codec = _codec_config(polarity, timestep, dim)
    core_config = dict(_core_config(), polarity=polarity, timestep=timestep, generator='sobol')
    encoder = encode(dict(codec)).to(device)
    core = conv_ugemm(weight.clone(), bias.clone(), stride=1, padding=PADDING, dilation=1,
                      config=core_config).to(device)
    decoder = decode(dict(codec)).to(device)
    return encoder, core, decoder


def _run(operation, input_value, timesteps=TIMESTEP):
    output = None
    for _ in range(timesteps):
        output = operation(input_value)
    return output


def test_conv_ugemm_hub_fidelity():
    """Verify the progressive output reaches (conv2d(x, W) + b) / entry on every device."""
    for device in devices():
        for polarity in POLARITIES:
            weight, bias = _parameters(polarity)
            input_value = _input_value(polarity).to(device)
            reference = F.conv2d(input_value, weight.to(device), bias.to(device),
                                 stride=1, padding=PADDING) / ENTRY

            layer = _make_hub(polarity, device)
            # A colliding encoder dim is numerically invisible: Sobol prefixes of length
            # 2**k are equidistributed in every dimension, so the full-period output is
            # bit-identical to the correct run, and mid-run the rmse moves only a few
            # percent, inside the noise. Comparing the two number sequences is the only
            # cover that the input stream really decorrelates from the weight stream.
            assert not torch.equal(layer.reference_encode.num_seq, layer.core.mul.num_seq)
            output = _run(layer, input_value)
            rmse = (output - reference).pow(2).mean().sqrt().item()

            assert output.shape == (BATCH, OUT_CHANNELS, SIDE, SIDE), output.shape
            assert layer.timestep_cur == TIMESTEP
            assert layer.core.timestep_cur == TIMESTEP
            assert layer.decoder.timestep_cur == TIMESTEP
            bound = RMSE_BOUND[polarity]
            assert rmse <= bound, (
                f'[{device}][{polarity}] rmse={rmse:.6f} exceeds bound={bound:.6f}'
            )
            print(f'[{device}][{polarity}] N={TIMESTEP}, rmse={rmse:.6f}, '
                  f'bound={bound:.6f}')


def test_conv_ugemm_hub_matches_bare_composition():
    """Verify the layer is bit-exact against a manual encode, bare core, and decode chain."""
    for device in devices():
        for polarity in POLARITIES:
            input_value = _input_value(polarity).to(device)
            layer = _make_hub(polarity, device)
            encoder, core, decoder = _make_bare(polarity, device)

            for _ in range(TIMESTEP):
                hub_output = layer(input_value)
                decoder(core(encoder(input_value)))
                bare_output = decoder.spike_value
                assert torch.equal(hub_output, bare_output), (
                    f'[{device}][{polarity}] diverged at timestep {layer.timestep_cur}'
                )
            print(f'[{device}][{polarity}] bit-exact against the bare composition '
                  f'for {TIMESTEP} timesteps.')


def test_conv_ugemm_hub_reset_replay():
    """Verify reset clears the run and an identical replay reproduces every output bit-exactly."""
    timesteps = 64
    for device in devices():
        for polarity in POLARITIES:
            input_value = _input_value(polarity).to(device)
            layer = _make_hub(polarity, device, timestep=timesteps)

            first = [layer(input_value).clone() for _ in range(timesteps)]
            assert layer.timestep_cur == timesteps
            layer.reset()
            assert layer.timestep_cur == 0
            assert layer.core.timestep_cur == 0
            assert layer.reference_encode.timestep_cur == 0
            assert layer.decoder.timestep_cur == 0
            assert layer.decoder.spike_count.abs().sum().item() == 0
            assert layer.core._im2col_idx is None

            replay = [layer(input_value).clone() for _ in range(timesteps)]
            assert all(torch.equal(before, after) for before, after in zip(first, replay))
            print(f'[{device}][{polarity}] reset and replay reproduced {timesteps} outputs.')


def test_conv_ugemm_hub_rejects_invalid_config():
    """Verify construction rejects codec and core configuration errors with exact messages."""
    weight, bias = _parameters('bipolar')
    accepted = "<['dim', 'generator', 'name', 'polarity', 'seed', 'taps', 'timestep']>"
    cases = [
        (_codec_config('bipolar'), dict(_core_config(), generator='sobol'),
         'Invalid key <generator> in the conv_ugemm_hub core configuration; the codec '
         'configuration supplies <polarity>, <timestep>, and <generator>.'),
        (_codec_config('bipolar', dim=1), _core_config(),
         'Invalid dim: <1>; legal values: any dimension other than the core dim <1>, so the '
         'input stream decorrelates from the weight stream.'),
        ({'polarity': 'bipolar', 'generator': 'sobol'}, _core_config(),
         'Missing key <timestep> in the input configuration.'),
        ({'timestep': TIMESTEP, 'generator': 'sobol'}, _core_config(),
         'Missing key <polarity> in the input configuration.'),
        (dict(_codec_config('bipolar'), scale=2), _core_config(),
         f'Unknown key <scale> in the input configuration; accepted keys: {accepted}.'),
    ]
    for codec_config, core_config, message in cases:
        try:
            conv_ugemm_hub(weight.clone(), bias.clone(), 1, PADDING, 1,
                           codec_config, core_config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(
                f'conv_ugemm_hub accepted the invalid config <{codec_config}> '
                f'and <{core_config}>.'
            )
    print('Test passed.')


def test_conv_ugemm_hub_performance():
    """Measure one timestep per device on a shared input of about 1e5 elements."""
    batch = 1500
    count = batch * IN_CHANNELS * SIDE * SIDE
    performance_input = torch.linspace(-1.0, 1.0, count, dtype=global_config.ntype).view(
        batch, IN_CHANNELS, SIDE, SIDE)
    assert performance_input.numel() >= 1e5
    cpu_runtime = None
    for device in devices():
        layer = _make_hub('bipolar', device)
        device_runtime = benchmark(
            lambda inputs: layer(inputs[0]),
            (performance_input,),
            device,
            warmup_runs=2,
            trials=7,
            prepare=layer.reset,
        )
        if device == 'cpu':
            cpu_runtime = device_runtime
        assert cpu_runtime is not None
        print(f'[{device}] device_runtime={device_runtime * 1e3:.3f}ms, '
              f'cpu_runtime={cpu_runtime * 1e3:.3f}ms, warmup=2, trials=7, median, '
              f'speedup={cpu_runtime / device_runtime:.2f}x')


if __name__ == '__main__':
    test_conv_ugemm_hub_rejects_invalid_config()
    test_conv_ugemm_hub_fidelity()
    test_conv_ugemm_hub_matches_bare_composition()
    test_conv_ugemm_hub_reset_replay()
    test_conv_ugemm_hub_performance()
    print('Test passed.')

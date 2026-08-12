"""Test the hybrid unary-binary linear layer.

Gradients are exempt: the layer wraps ``linear_ugemm``, whose spike comparison is
not differentiable and carries no straight-through estimator, so no gradient ever
reaches its weight and bias parameters and gate 6 has no defining equation to
check.
"""

import torch
import torch.nn.functional as F

from napl.sim.base import global_config
from napl.sim.module import linear_ugemm, linear_ugemm_hub
from napl.sim.operation import decode, encode
from napl.utils._shared_test import benchmark, devices


TIMESTEP = 512
BATCH, IN_FEATURES, OUT_FEATURES = 3, 6, 4
ENTRY = IN_FEATURES + 1
POLARITIES = ['unipolar', 'bipolar']


def _codec_config(polarity, timestep=TIMESTEP, dim=2):
    return {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': dim}


def _core_config():
    return {'dim': 1, 'scale': None, 'width': 12}


def _parameters(polarity):
    weight = torch.linspace(-0.5, 0.5, OUT_FEATURES * IN_FEATURES,
                            dtype=global_config.ntype).view(OUT_FEATURES, IN_FEATURES)
    bias = torch.linspace(0.4, -0.4, OUT_FEATURES, dtype=global_config.ntype)
    if polarity == 'unipolar':
        weight = weight.abs()
        bias = bias.abs()
    return weight, bias


def _input_value(polarity):
    low = 0.0 if polarity == 'unipolar' else -1.0
    return torch.linspace(low, 1.0, BATCH * IN_FEATURES,
                          dtype=global_config.ntype).view(BATCH, IN_FEATURES)


def _make_hub(polarity, device, timestep=TIMESTEP, dim=2):
    weight, bias = _parameters(polarity)
    return linear_ugemm_hub(weight.clone(), bias.clone(),
                            _codec_config(polarity, timestep, dim), _core_config()).to(device)


def _make_bare(polarity, device, timestep=TIMESTEP, dim=2):
    """Return the manual encode, bare core, and decode composition the layer wraps."""
    weight, bias = _parameters(polarity)
    codec = _codec_config(polarity, timestep, dim)
    core_config = dict(_core_config(), polarity=polarity, timestep=timestep, generator='sobol')
    encoder = encode(dict(codec)).to(device)
    core = linear_ugemm(weight.clone(), bias.clone(), config=core_config).to(device)
    decoder = decode(dict(codec)).to(device)
    return encoder, core, decoder


def _step_run(operation, input_value, timesteps=TIMESTEP):
    """Return the final output of an explicit per-timestep run."""
    output = None
    for _ in range(timesteps):
        output = operation.forward_timestep(input_value)
    return output


def test_linear_ugemm_hub_fidelity():
    """Verify the progressive output reaches (Wx + b) / entry for both polarities on every device."""
    for device in devices():
        for polarity in POLARITIES:
            weight, bias = _parameters(polarity)
            input_value = _input_value(polarity).to(device)
            reference = F.linear(input_value, weight.to(device), bias.to(device)) / ENTRY

            layer = _make_hub(polarity, device)
            # A colliding encoder dim is numerically invisible: Sobol prefixes of length
            # 2**k are equidistributed in every dimension, so the full-period output is
            # bit-identical to the correct run, and mid-run the rmse moves only a few
            # percent, inside the noise. Comparing the two number sequences is the only
            # cover that the input stream really decorrelates from the weight stream.
            assert not torch.equal(layer.reference_encode.num_seq, layer.core.mul.num_seq)
            output = layer(input_value)
            rmse = (output - reference).pow(2).mean().sqrt().item()

            assert output.shape == (BATCH, OUT_FEATURES), output.shape
            # The layer is non-streaming at its numeric interface, so its own
            # counter stays at 0 while the streaming parts it drives count the
            # whole run.
            assert layer.streaming is False
            assert layer.timestep_cur == 0
            assert layer.core.timestep_cur == TIMESTEP
            assert layer.decoder.timestep_cur == TIMESTEP
            # The core multiplier turns the weights into streams itself.
            assert layer.internal_encode is True
            print(f'[{device}][{polarity}] N={TIMESTEP}, rmse={rmse:.6f}')


def test_linear_ugemm_hub_matches_bare_composition():
    """Verify the layer is bit-exact against a manual encode, bare core, and decode chain."""
    for device in devices():
        for polarity in POLARITIES:
            input_value = _input_value(polarity).to(device)
            layer = _make_hub(polarity, device)
            encoder, core, decoder = _make_bare(polarity, device)

            for _ in range(TIMESTEP):
                hub_output = layer.forward_timestep(input_value)
                decoder(core(encoder(input_value)))
                bare_output = decoder.spike_value
                assert torch.equal(hub_output, bare_output), (
                    f'[{device}][{polarity}] diverged at timestep {layer.core.timestep_cur}'
                )
            print(f'[{device}][{polarity}] bit-exact against the bare composition '
                  f'for {TIMESTEP} timesteps.')


def test_linear_ugemm_hub_reset_replay():
    """Verify reset clears the run and an identical replay reproduces every output bit-exactly."""
    timesteps = 64
    for device in devices():
        for polarity in POLARITIES:
            input_value = _input_value(polarity).to(device)
            layer = _make_hub(polarity, device, timestep=timesteps)

            first = [layer.forward_timestep(input_value).clone() for _ in range(timesteps)]
            assert layer.core.timestep_cur == timesteps
            layer.reset()
            assert layer.timestep_cur == 0
            assert layer.core.timestep_cur == 0
            assert layer.reference_encode.timestep_cur == 0
            assert layer.decoder.timestep_cur == 0
            assert layer.decoder.spike_count.abs().sum().item() == 0

            replay = [layer.forward_timestep(input_value).clone() for _ in range(timesteps)]
            assert all(torch.equal(before, after) for before, after in zip(first, replay))
            print(f'[{device}][{polarity}] reset and replay reproduced {timesteps} outputs.')


def test_linear_ugemm_hub_matches_timestep_loop():
    """Verify one decorated call equals an explicit timestep loop and repeats bit-exactly."""
    for device in devices():
        for polarity in POLARITIES:
            input_value = _input_value(polarity).to(device)

            looped_layer = _make_hub(polarity, device)
            looped = _step_run(looped_layer, input_value)
            assert looped_layer.core.timestep_cur == TIMESTEP

            layer = _make_hub(polarity, device)
            single = layer(input_value)
            assert torch.equal(single, looped), (
                f'[{device}][{polarity}] the decorated call and the {TIMESTEP}-timestep loop '
                f'disagree'
            )
            assert layer.timestep_cur == 0
            assert layer.core.timestep_cur == TIMESTEP
            assert layer.decoder.timestep_cur == TIMESTEP

            # A decorated call is a fresh run, so a partly advanced layer decodes
            # the same value as an untouched one.
            advanced_layer = _make_hub(polarity, device)
            advanced_layer.forward_timestep(input_value)
            assert torch.equal(advanced_layer(input_value), single), (
                f'[{device}][{polarity}] a repeated decorated call is not a fresh run'
            )
            print(f'[{device}][{polarity}] one call matches the {TIMESTEP}-timestep loop '
                  f'bit-exactly.')


def test_linear_ugemm_hub_progressive_precision():
    """Verify per-timestep stepping refines the output toward the analytic result."""
    checkpoints = (TIMESTEP // 8, TIMESTEP)
    for device in devices():
        for polarity in POLARITIES:
            weight, bias = _parameters(polarity)
            input_value = _input_value(polarity).to(device)
            reference = F.linear(input_value, weight.to(device), bias.to(device)) / ENTRY

            layer = _make_hub(polarity, device)
            errors = []
            for timestep in range(1, TIMESTEP + 1):
                output = layer.forward_timestep(input_value)
                assert layer.core.timestep_cur == timestep
                assert output.shape == (BATCH, OUT_FEATURES), output.shape
                if timestep in checkpoints:
                    errors.append((output - reference).pow(2).mean().sqrt().item())
            assert errors[-1] < errors[0], errors
            print(f'[{device}][{polarity}] rmse refines from {errors[0]:.6f} at '
                  f'N={checkpoints[0]} to {errors[-1]:.6f} at N={checkpoints[1]}.')


def test_linear_ugemm_hub_rejects_invalid_config():
    """Verify construction rejects codec and core configuration errors with exact messages."""
    weight, bias = _parameters('bipolar')
    accepted = "<['dim', 'generator', 'name', 'polarity', 'seed', 'taps', 'timestep']>"
    cases = [
        (_codec_config('bipolar'), dict(_core_config(), timestep=TIMESTEP),
         'Invalid key <timestep> in the linear_ugemm_hub core configuration; the codec '
         'configuration supplies <polarity>, <timestep>, and <generator>.'),
        (_codec_config('bipolar'), dict(_core_config(), polarity='bipolar'),
         'Invalid key <polarity> in the linear_ugemm_hub core configuration; the codec '
         'configuration supplies <polarity>, <timestep>, and <generator>.'),
        (_codec_config('bipolar', dim=1), _core_config(),
         'Invalid dim: <1>; legal values: any dimension other than the core dim <1>, so the '
         'input stream decorrelates from the weight stream.'),
        ({'polarity': 'bipolar', 'timestep': TIMESTEP}, _core_config(),
         'Missing key <generator> in the input configuration.'),
        ({'timestep': TIMESTEP, 'generator': 'sobol'}, _core_config(),
         'Missing key <polarity> in the input configuration.'),
        (dict(_codec_config('bipolar'), width=8), _core_config(),
         f'Unknown key <width> in the input configuration; accepted keys: {accepted}.'),
    ]
    for codec_config, core_config, message in cases:
        try:
            linear_ugemm_hub(weight.clone(), bias.clone(), codec_config, core_config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(
                f'linear_ugemm_hub accepted the invalid config <{codec_config}> '
                f'and <{core_config}>.'
            )
    print('Test passed.')


def test_linear_ugemm_hub_performance():
    """Measure one timestep per device on a shared input of about 1e5 elements."""
    rows = 20000
    performance_input = torch.linspace(-1.0, 1.0, rows * IN_FEATURES,
                                       dtype=global_config.ntype).view(rows, IN_FEATURES)
    assert performance_input.numel() >= 1e5
    cpu_runtime = None
    for device in devices():
        layer = _make_hub('bipolar', device)
        device_runtime = benchmark(
            lambda inputs: layer.forward_timestep(inputs[0]),
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
    test_linear_ugemm_hub_rejects_invalid_config()
    test_linear_ugemm_hub_fidelity()
    test_linear_ugemm_hub_matches_bare_composition()
    test_linear_ugemm_hub_matches_timestep_loop()
    test_linear_ugemm_hub_progressive_precision()
    test_linear_ugemm_hub_reset_replay()
    test_linear_ugemm_hub_performance()
    print('Test passed.')

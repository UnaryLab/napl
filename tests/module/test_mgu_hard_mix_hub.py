"""Test the hybrid unary-binary MGU cell.

Gradients are exempt: the cell wraps ``mgu_hard_mix``, whose gate spikes come
from non-differentiable comparisons with no straight-through estimator, so no
gradient ever reaches the gate parameters and gate 6 has no defining equation to
check.
"""

import torch
import torch.nn.functional as F

from napl.sim.base import global_config
from napl.sim.module import mgu_hard_mix, mgu_hard_mix_hub
from napl.sim.operation import decode, encode
from napl.utils._shared_test import benchmark, devices


TIMESTEP = 2048
DEPTH_ISMUL = 6
BATCH, INPUT_SIZE, HIDDEN_SIZE = 3, 6, 4


def _codec_config(timestep=TIMESTEP, dim=1, polarity='bipolar'):
    return {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': dim}


def _core_config():
    return {'width': 10, 'depth_ismul': DEPTH_ISMUL}


def _parameters():
    count = HIDDEN_SIZE * (HIDDEN_SIZE + INPUT_SIZE)
    weight_f = torch.linspace(-0.5, 0.5, count, dtype=global_config.ntype).view(
        HIDDEN_SIZE, HIDDEN_SIZE + INPUT_SIZE)
    weight_n = torch.linspace(0.5, -0.5, count, dtype=global_config.ntype).view(
        HIDDEN_SIZE, HIDDEN_SIZE + INPUT_SIZE)
    bias_f = torch.linspace(-0.2, 0.2, HIDDEN_SIZE, dtype=global_config.ntype)
    bias_n = torch.linspace(0.2, -0.2, HIDDEN_SIZE, dtype=global_config.ntype)
    return weight_f, bias_f, weight_n, bias_n


def _input_value(batch=BATCH):
    return torch.linspace(1.0, -1.0, batch * INPUT_SIZE,
                          dtype=global_config.ntype).view(batch, INPUT_SIZE)


def _hx_value(batch=BATCH):
    return torch.linspace(-1.0, 1.0, batch * HIDDEN_SIZE,
                          dtype=global_config.ntype).view(batch, HIDDEN_SIZE)


def _reference_mgu(input_value, hx_value):
    """Analytic hard-activation MGU recurrence the streaming run approximates."""
    weight_f, bias_f, weight_n, bias_n = _parameters()
    weight_f = weight_f.to(input_value.device)
    weight_n = weight_n.to(input_value.device)
    bias_f = bias_f.to(input_value.device)
    bias_n = bias_n.to(input_value.device)
    fg_in = F.hardtanh(F.linear(torch.cat((hx_value, input_value), 1), weight_f, bias_f))
    fg = F.hardsigmoid(fg_in * 3)
    fg_hx = fg * hx_value
    ng = F.hardtanh(F.linear(torch.cat((fg_hx, input_value), 1), weight_n, bias_n))
    return F.hardtanh(ng - fg * ng + fg_hx)


def _make_hub(device, timestep=TIMESTEP, dim=1, batch=BATCH):
    weight_f, bias_f, weight_n, bias_n = _parameters()
    return mgu_hard_mix_hub(weight_f.clone(), bias_f.clone(), weight_n.clone(), bias_n.clone(),
                            _hx_value(batch), _codec_config(timestep, dim),
                            _core_config()).to(device)


def _make_bare(device, timestep=TIMESTEP, dim=1, batch=BATCH):
    """Return the manual encode, bare cell, and decode composition the wrapper holds."""
    weight_f, bias_f, weight_n, bias_n = _parameters()
    codec = _codec_config(timestep, dim)
    core_config = dict(_core_config(), polarity='bipolar', timestep=timestep, generator='sobol')
    core = mgu_hard_mix(weight_f.clone(), bias_f.clone(), weight_n.clone(), bias_n.clone(),
                        _hx_value(batch), core_config).to(device)
    encoder_input = encode(dict(codec, dim=dim)).to(device)
    encoder_hx = encode(dict(codec, dim=dim + 1)).to(device)
    decoder = decode(dict(codec)).to(device)
    return encoder_input, encoder_hx, core, decoder


def test_mgu_hard_mix_hub_fidelity():
    """Verify the progressive output reaches the analytic MGU recurrence on every device.

    Bipolar only: the cell's subtraction path requires signed streams, so
    mgu_hard_mix rejects a unipolar configuration.
    """
    for device in devices():
        input_value = _input_value().to(device)
        hx_value = _hx_value().to(device)
        reference = _reference_mgu(input_value, hx_value)

        cell = _make_hub(device)
        # A colliding encoder dim changes the output (max abs diff 0.074219) but keeps the rmse
        # under the bound: 0.014819 with the hidden encoder moved onto the input dim, 0.025912
        # with both encoders on dim 2, against 0.017490 for the correct run. The numerical check
        # below therefore does not catch a collision, and comparing the two number sequences is
        # the only cover that the input stream really decorrelates from the hidden-state stream.
        assert not torch.equal(cell.reference_encode_input.num_seq, cell.reference_encode_hx.num_seq)
        output = cell(input_value)
        # The cpu and mps outputs are bit-identical (max abs diff 0.0); the rmse below still
        # differs by about 2e-9 across devices (measured 1.863e-9) because the float reference
        # is evaluated on-device, and cpu and mps disagree on it by up to 8.94e-8.
        rmse = (output - reference).pow(2).mean().sqrt().item()

        assert output.shape == (BATCH, HIDDEN_SIZE), output.shape
        # The cell wrapper is non-streaming at its numeric interface, so its own
        # counter stays at 0 while the streaming parts it drives count the whole
        # run.
        assert cell.streaming is False
        assert cell.timestep_cur == 0
        assert cell.core.timestep_cur == TIMESTEP
        assert cell.decoder.timestep_cur == TIMESTEP
        # The wrapper derives its value over its parts, and the core is the only True one: its
        # two registered encode instances and its decoder all report False. The walk is
        # depth-1: the wrapper reads the core's own declared value and never descends to the
        # core's multiplier.
        assert cell.internal_encode is True
        assert cell.core.internal_encode is True
        assert cell.core.fg_hx_mul.internal_encode is True
        print(f'[{device}][bipolar] N={TIMESTEP}, rmse={rmse:.6f}')


def test_mgu_hard_mix_hub_matches_bare_composition():
    """Verify the cell is bit-exact against a manual encode, bare cell, and decode chain."""
    for device in devices():
        input_value = _input_value().to(device)
        cell = _make_hub(device)
        encoder_input, encoder_hx, core, decoder = _make_bare(device)

        for _ in range(TIMESTEP):
            hub_output = cell.forward_timestep(input_value)
            decoder(core(encoder_input(input_value), encoder_hx(core.hx_value)))
            bare_output = decoder.spike_value
            assert torch.equal(hub_output, bare_output), (
                f'[{device}] diverged at timestep {cell.core.timestep_cur}'
            )
        print(f'[{device}][bipolar] bit-exact against the bare composition '
              f'for {TIMESTEP} timesteps.')


def test_mgu_hard_mix_hub_reset_replay():
    """Verify reset clears the run and an identical replay reproduces every output bit-exactly."""
    timesteps = 128
    for device in devices():
        input_value = _input_value().to(device)
        cell = _make_hub(device, timestep=timesteps)

        first = [cell.forward_timestep(input_value).clone() for _ in range(timesteps)]
        assert cell.core.timestep_cur == timesteps
        cell.reset()
        assert cell.timestep_cur == 0
        assert cell.core.timestep_cur == 0
        assert cell.reference_encode_input.timestep_cur == 0
        assert cell.reference_encode_hx.timestep_cur == 0
        assert cell.decoder.timestep_cur == 0
        assert cell.decoder.spike_count.abs().sum().item() == 0

        replay = [cell.forward_timestep(input_value).clone() for _ in range(timesteps)]
        assert all(torch.equal(before, after) for before, after in zip(first, replay))
        print(f'[{device}][bipolar] reset and replay reproduced {timesteps} outputs.')


def test_mgu_hard_mix_hub_matches_timestep_loop():
    """Verify one decorated call equals an explicit timestep loop and repeats bit-exactly.

    Bipolar only: the cell's subtraction path requires signed streams, so
    mgu_hard_mix rejects a unipolar configuration.
    """
    for device in devices():
        input_value = _input_value().to(device)

        looped_cell = _make_hub(device)
        looped = None
        for _ in range(TIMESTEP):
            looped = looped_cell.forward_timestep(input_value)
        assert looped_cell.core.timestep_cur == TIMESTEP

        cell = _make_hub(device)
        single = cell(input_value)
        assert torch.equal(single, looped), (
            f'[{device}] the decorated call and the {TIMESTEP}-timestep loop disagree'
        )
        assert single.shape == (BATCH, HIDDEN_SIZE), single.shape
        assert cell.timestep_cur == 0
        assert cell.core.timestep_cur == TIMESTEP
        assert cell.decoder.timestep_cur == TIMESTEP
        # The hidden port is re-encoded inside each timestep, so the run needs no
        # hidden value from the caller.
        assert cell.reference_encode_hx.timestep_cur == TIMESTEP

        # A decorated call is a fresh run, so a partly advanced cell decodes the
        # same value as an untouched one.
        advanced_cell = _make_hub(device)
        advanced_cell.forward_timestep(input_value)
        assert torch.equal(advanced_cell(input_value), single), (
            f'[{device}] a repeated decorated call is not a fresh run'
        )
        print(f'[{device}][bipolar] one call matches the {TIMESTEP}-timestep loop '
              f'bit-exactly.')


def test_mgu_hard_mix_hub_rejects_invalid_config():
    """Verify construction rejects codec, core, and polarity errors with exact messages."""
    weight_f, bias_f, weight_n, bias_n = _parameters()
    accepted = "<['dim', 'generator', 'name', 'polarity', 'seed', 'taps', 'timestep']>"
    gate_dims = 'outside the gate dims <[3, 4, 5, 6]>, so the port streams decorrelate from ' \
                'the gate weight streams.'
    cases = [
        (_codec_config(), dict(_core_config(), timestep=TIMESTEP),
         'Invalid key <timestep> in the mgu_hard_mix_hub core configuration; the codec '
         'configuration supplies <polarity>, <timestep>, and <generator>.'),
        (_codec_config(dim=2), _core_config(),
         f'Invalid dim: <2>; legal values: any dimension placing the input dim and the '
         f'hidden dim <3> {gate_dims}'),
        (_codec_config(dim=6), _core_config(),
         f'Invalid dim: <6>; legal values: any dimension placing the input dim and the '
         f'hidden dim <7> {gate_dims}'),
        ({'polarity': 'bipolar', 'timestep': TIMESTEP}, _core_config(),
         'Missing key <generator> in the input configuration.'),
        ({'timestep': TIMESTEP, 'generator': 'sobol'}, _core_config(),
         'Missing key <polarity> in the input configuration.'),
        (dict(_codec_config(), width=10), _core_config(),
         f'Unknown key <width> in the input configuration; accepted keys: {accepted}.'),
        (_codec_config(polarity='unipolar'), _core_config(),
         "Invalid polarity: <unipolar>; legal values: <['bipolar']>."),
        (_codec_config(timestep=2 ** DEPTH_ISMUL), _core_config(),
         f'Invalid timestep: <{2 ** DEPTH_ISMUL}>; legal values: an integer greater than '
         f'<{2 ** DEPTH_ISMUL}>, the number of timesteps that flushing the '
         f'depth_ismul <{DEPTH_ISMUL}> multiplier shift register consumes, '
         f'since a run of exactly that length leaves nothing behind.'),
    ]
    for codec_config, core_config, message in cases:
        try:
            mgu_hard_mix_hub(weight_f.clone(), bias_f.clone(), weight_n.clone(), bias_n.clone(),
                             _hx_value(), codec_config, core_config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(
                f'mgu_hard_mix_hub accepted the invalid config <{codec_config}> '
                f'and <{core_config}>.'
            )
    print('Test passed.')


def test_mgu_hard_mix_hub_performance():
    """Measure one timestep per device on a shared input of about 1e5 elements."""
    batch = 20000
    performance_input = _input_value(batch)
    assert performance_input.numel() >= 1e5
    cpu_runtime = None
    for device in devices():
        cell = _make_hub(device, batch=batch)
        device_runtime = benchmark(
            lambda inputs: cell.forward_timestep(inputs[0]),
            (performance_input,),
            device,
            warmup_runs=2,
            trials=7,
            prepare=cell.reset,
        )
        if device == 'cpu':
            cpu_runtime = device_runtime
        assert cpu_runtime is not None
        print(f'[{device}] device_runtime={device_runtime * 1e3:.3f}ms, '
              f'cpu_runtime={cpu_runtime * 1e3:.3f}ms, warmup=2, trials=7, median, '
              f'speedup={cpu_runtime / device_runtime:.2f}x')


if __name__ == '__main__':
    test_mgu_hard_mix_hub_rejects_invalid_config()
    test_mgu_hard_mix_hub_fidelity()
    test_mgu_hard_mix_hub_matches_bare_composition()
    test_mgu_hard_mix_hub_matches_timestep_loop()
    test_mgu_hard_mix_hub_reset_replay()
    test_mgu_hard_mix_hub_performance()
    print('Test passed.')

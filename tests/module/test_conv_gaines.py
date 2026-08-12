import torch
import torch.nn.functional as F

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.module import conv_gaines, conv_mix, linear_gaines
from napl.sim.operation import encode, decode


class napl_conv_gaines(napl_base):
    """Wire encoder -> conv_gaines -> decode (canonical round-trip)."""


    def __init__(self, codec_config, weight, bias, stride, padding, conv_config):
        super().__init__()
        self.encoder = encode(codec_config)
        self.decoder = decode(codec_config)
        self.conv = conv_gaines(weight, bias, stride=stride, padding=padding,
                                config=conv_config)


    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        self.decoder(self.conv(self.encoder(input_x)))


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.encoder.reset()
        self.conv.reset()
        self.decoder.reset()


def _scaled_reference(input_x, weight, bias, padding, entry, polarity):
    """Value the scaled Gaines conv converges to: conv2d(x, W) + b divided by entry."""
    low, high = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    reference = F.conv2d(input_x, weight, bias, stride=1, padding=padding) / entry
    return reference.clamp(low, high)


def _fidelity_checks():
    """Scaled conv_gaines tracks (conv2d(x, W) + b) / entry for both polarities.

    The residual error is systematic rather than stochastic and does not shrink
    with timestep: the Gaines MUX selects one weight stream per timestep on a
    period-``entry`` schedule, and a Sobol threshold sequence is not balanced on
    that sub-lattice, so each product contributes a slightly biased rate. Small
    ``entry`` makes this worse, which is why these cases use ``entry = 16``.

    The check runs the kernel for both polarities and paddings and prints the
    measured rmse for inspection; it asserts no fidelity bound. Measured
    unipolar rmse at timestep 1024 is about 0.047, and bipolar rmse is about
    0.131 at padding 0 and 0.105 to 0.112 at padding 1.
    """
    ntype = global_config.ntype
    timestep = 1024
    batch, out_channels, size = 2, 4, 8
    torch.manual_seed(42)  # Keep gen_rand_tensor inputs reproducible.

    # Draw every random input on CPU, outside the device loop.
    cases = []
    for polarity in ['unipolar', 'bipolar']:
        for has_bias in [True, False]:
            # entry = in_channels * kh * kw + has_bias must be a power of two.
            in_channels, k = (15, 1) if has_bias else (4, 2)
            input_x_cpu = gen_rand_tensor(
                polarity, (batch, in_channels, size, size), 8).type(ntype)
            weight_cpu = gen_rand_tensor(
                polarity, (out_channels, in_channels, k, k), 8).type(ntype)
            bias_cpu = gen_rand_tensor(
                polarity, (out_channels,), 8).type(ntype) if has_bias else None
            cases.append((polarity, has_bias, in_channels, k,
                          input_x_cpu, weight_cpu, bias_cpu))

    for device in devices():
        for polarity, has_bias, in_channels, k, input_x_cpu, weight_cpu, bias_cpu in cases:
            for padding in [0, 1]:
                input_x = input_x_cpu.to(device)
                weight = weight_cpu.to(device)
                bias = None if bias_cpu is None else bias_cpu.to(device)
                entry = in_channels * k * k + (1 if has_bias else 0)

                codec_config = {'polarity': polarity, 'timestep': timestep,
                                'generator': 'sobol', 'dim': 1}
                conv_config = {'polarity': polarity, 'timestep': timestep,
                               'generator': 'sobol', 'dim': 2, 'scaled': True}
                inst = napl_conv_gaines(codec_config, weight, bias, 1, padding,
                                        conv_config).to(device)
                inst(input_x, timesteps=timestep)

                reference = _scaled_reference(input_x, weight, bias, padding, entry, polarity)
                error = (inst.decoder.spike_value - reference).abs()
                rmse = error.pow(2).mean().sqrt().item()
                assert inst.decoder.spike_value.shape == reference.shape, \
                    (device, polarity, has_bias, padding)
                assert inst.conv.timestep_cur == timestep
                # The layer holds its own weight and bias encoders.
                assert inst.conv.internal_encode is True
                print(f'[{device}] {polarity} bias={has_bias} entry={entry} pad={padding}: '
                      f'rmse={rmse:.5f} max_err={error.max().item():.5f}')
                inst.reset()


def _geometry_equivalence_check():
    """conv_gaines equals a bare linear_gaines fed the same im2col patches, bit for bit.

    Both paths share the Gaines core construction, so the core's systematic rate
    bias cancels and only the im2col indices, stride, padding, dilation, and the
    output reshape are under test. Bipolar padding is the one case where full
    equality does not apply: the conv owns a decorrelated rate-0.5 pad stream
    that the bare linear path has no counterpart for, so the comparison is
    restricted to the interior windows the padding never reaches.
    """
    ntype = global_config.ntype
    timestep = 8  # Exact equality, so a short stream is enough.
    batch, out_channels, size = 2, 4, 8
    torch.manual_seed(42)  # Keep gen_rand_tensor inputs reproducible.

    # Draw every random input on CPU, outside the device loop.
    cases = []
    for polarity in ['unipolar', 'bipolar']:
        for has_bias in [True, False]:
            # entry = in_channels * kh * kw + has_bias must be a power of two.
            in_channels, k = (15, 1) if has_bias else (4, 2)
            input_x_cpu = gen_rand_tensor(
                polarity, (batch, in_channels, size, size), 8).type(ntype)
            weight_cpu = gen_rand_tensor(
                polarity, (out_channels, in_channels, k, k), 8).type(ntype)
            bias_cpu = gen_rand_tensor(
                polarity, (out_channels,), 8).type(ntype) if has_bias else None
            cases.append((polarity, has_bias, in_channels, k,
                          input_x_cpu, weight_cpu, bias_cpu))

    for device in devices():
        checked = 0
        for polarity, has_bias, in_channels, k, input_x_cpu, weight_cpu, bias_cpu in cases:
            for stride in [1, 2]:
                for dilation in [1, 2]:
                    for padding in [0, 1]:
                        input_x = input_x_cpu.to(device)
                        weight = weight_cpu.to(device)
                        bias = None if bias_cpu is None else bias_cpu.to(device)
                        conv_config = {'polarity': polarity, 'timestep': timestep,
                                       'generator': 'sobol', 'dim': 2, 'scaled': True}
                        conv = conv_gaines(weight, bias, stride=stride, padding=padding,
                                           dilation=dilation, config=conv_config).to(device)
                        core = linear_gaines(weight.reshape(out_channels, -1), bias,
                                             config=conv_config).to(device)
                        encoder = encode({'polarity': polarity, 'timestep': timestep,
                                          'generator': 'sobol', 'dim': 1}).to(device)
                        fan_in = in_channels * k * k

                        # Interior output window range the padding never touches.
                        span = dilation * (k - 1) + 1
                        low = -(-padding // stride)
                        high = (size + padding - span) // stride
                        interior = polarity == 'bipolar' and padding != 0
                        assert high >= low, (device, polarity, stride, dilation, padding)

                        for _ in range(timestep):
                            input_spike = encoder(input_x)
                            conv_spike = conv(input_spike)
                            patches = F.unfold(input_spike.type(ntype), (k, k),
                                               dilation, padding, stride)
                            core_spike = core(patches.transpose(1, 2).reshape(-1, fan_in))
                            core_spike = core_spike.view(batch, -1, out_channels) \
                                                   .transpose(1, 2).reshape(conv_spike.shape)
                            if interior:
                                conv_spike = conv_spike[..., low:high + 1, low:high + 1]
                                core_spike = core_spike[..., low:high + 1, low:high + 1]
                            assert torch.equal(conv_spike, core_spike), \
                                f'[{device}] {polarity} bias={has_bias} stride={stride} ' \
                                f'dilation={dilation} pad={padding}: conv_gaines geometry ' \
                                f'diverged from the bare linear_gaines path'
                        checked += 1
                        conv.reset()
                        core.reset()
                        encoder.reset()
        print(f'[{device}] {checked} conv_gaines geometries match linear_gaines bit-exactly '
              f'over {timestep} timesteps.')


def _pad_decorrelation_check():
    """The bipolar pad stream decodes to zero on an all-padding output border.

    A 1x1 kernel with padding=1 makes every border output position depend on
    padding alone, so its reference is exactly 0. The decorrelated rate-0.5 pad
    stream reaches a measured border rmse of 0.0039. A plain 0-pad injects
    bipolar -1 and lands at 0.447; the check asserts the plain 0-pad rmse
    exceeds 0.20, the discrimination threshold that separates the two.
    """
    ntype = global_config.ntype
    timestep = 1024
    torch.manual_seed(42)  # Keep gen_rand_tensor inputs reproducible.
    batch, in_channels, out_channels, size = 2, 2, 4, 4
    input_x_cpu = gen_rand_tensor('bipolar', (batch, in_channels, size, size), 8).type(ntype)
    weight_cpu = gen_rand_tensor('bipolar', (out_channels, in_channels, 1, 1), 8).type(ntype)

    codec_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
    conv_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol',
                   'dim': 2, 'scaled': True}
    for device in devices():
        input_x = input_x_cpu.to(device)
        weight = weight_cpu.to(device)
        reference = _scaled_reference(input_x, weight, None, 1, in_channels, 'bipolar')
        border = torch.ones_like(reference, dtype=torch.bool)
        border[..., 1:-1, 1:-1] = False

        rmse = {}
        for plain_zero_pad in [False, True]:
            inst = napl_conv_gaines(codec_config, weight, None, 1, 1, conv_config).to(device)
            if plain_zero_pad:
                # Mutant: a deterministic 0 pad bit decodes to bipolar -1.
                inst.conv.pad_bits = [0.0] * inst.conv.pad_len
            inst(input_x, timesteps=timestep)
            error = (inst.decoder.spike_value - reference).abs()
            rmse[plain_zero_pad] = error[border].pow(2).mean().sqrt().item()
            inst.reset()
        print(f'[{device}] border rmse: decorrelated pad={rmse[False]:.5f} '
              f'plain 0 pad={rmse[True]:.5f}')
        assert rmse[True] > 0.20, \
            f'[{device}] plain 0 pad should fail the border bound, got {rmse[True]}'


def _rejection_checks():
    """Illegal fan-in and non-scaled bipolar data are rejected at construction."""
    weight = torch.ones(2, 2, 2, 2).type(global_config.ntype)
    config = {'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol',
              'dim': 2, 'scaled': True}

    # entry = 2 * 2 * 2 + 1 = 9 is not a power of two.
    try:
        conv_gaines(weight, torch.zeros(2).type(global_config.ntype), config=config)
    except AssertionError as error:
        assert str(error) == 'conv_gaines scaled mode needs power-of-two entry, got 9.', error
    else:
        raise AssertionError('conv_gaines accepted a non-power-of-two entry')

    # A 1x1 kernel over one input channel leaves a single adder input.
    try:
        conv_gaines(torch.ones(2, 1, 1, 1).type(global_config.ntype), None, config=config)
    except AssertionError as error:
        assert str(error) == 'conv_gaines scaled mode needs entry >= 2, got 1.', error
    else:
        raise AssertionError('conv_gaines accepted entry < 2')

    # The composed core rejects non-scaled bipolar addition.
    try:
        conv_gaines(weight, None, config={**config, 'scaled': False})
    except AssertionError as error:
        assert str(error) == 'Non-scaled Gaines addition in conv_gaines does not support bipolar data.', error
    else:
        raise AssertionError('conv_gaines accepted non-scaled bipolar data')
    print('fan-in and non-scaled bipolar rejections raised as expected.')


def _known_answer_check():
    """All-ones unipolar operands make the scaled Gaines conv spike every timestep."""
    ntype = global_config.ntype
    in_channels, out_channels, k = 2, 3, 2
    weight_cpu = torch.ones(out_channels, in_channels, k, k).type(ntype)
    input_cpu = torch.ones(1, in_channels, k, k).type(ntype)
    for device in devices():
        layer = conv_gaines(
            weight_cpu.to(device), None, stride=1, padding=0,
            config={'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol',
                    'dim': 2, 'scaled': True},
        ).to(device)
        input_spike = input_cpu.to(device)
        for _ in range(64):
            output_spike = layer(input_spike)
            assert torch.equal(output_spike, torch.ones_like(output_spike)), \
                f'[{device}] all-ones unipolar Gaines conv should spike every timestep'
        layer.reset()
        print(f'[{device}] known-answer corner passed.')


def _non_scaled_check():
    """Non-scaled unipolar mode OR-reduces the products, staying between max and sum."""
    ntype = global_config.ntype
    timestep = 1024
    in_channels, out_channels, k = 2, 3, 2
    input_x_cpu = torch.full((1, in_channels, k, k), 0.25, dtype=ntype)
    weight_cpu = torch.full((out_channels, in_channels, k, k), 0.125, dtype=ntype)
    tolerance = 3.0 / timestep ** 0.5
    codec_config = {'polarity': 'unipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
    conv_config = {'polarity': 'unipolar', 'timestep': timestep, 'generator': 'sobol',
                   'dim': 2, 'scaled': False}
    for device in devices():
        input_x = input_x_cpu.to(device)
        weight = weight_cpu.to(device)
        inst = napl_conv_gaines(codec_config, weight, None, 1, 0, conv_config).to(device)
        inst(input_x, timesteps=timestep)
        low = (input_x * weight[0]).max()
        high = F.conv2d(input_x, weight)
        spike_value = inst.decoder.spike_value
        assert spike_value.shape == high.shape, (device, spike_value.shape, high.shape)
        assert high.max().item() < 1.0
        assert (spike_value >= low - tolerance).all(), \
            f'[{device}] non-scaled output {spike_value} below the largest product {low}'
        assert (spike_value <= high + tolerance).all(), \
            f'[{device}] non-scaled output {spike_value} above the product sum {high}'
        print(f'[{device}] non-scaled unipolar: rate={spike_value.min().item():.5f}..'
              f'{spike_value.max().item():.5f} in [{low.item():.5f}, {high.max().item():.5f}] '
              f'+/- {tolerance:.5f}')
        inst.reset()


def _reset_replay_check():
    """Replaying the same padded bipolar stream after reset reproduces the spikes."""
    ntype = global_config.ntype
    timestep = 64
    torch.manual_seed(42)  # Keep gen_rand_tensor inputs reproducible.
    weight_cpu = gen_rand_tensor('bipolar', (3, 2, 2, 2), 8).type(ntype)
    input_cpu = gen_rand_tensor('bipolar', (1, 2, 4, 4), 8).type(ntype)
    codec_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
    conv_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol',
                   'dim': 2, 'scaled': True}
    for device in devices():
        encoder = encode(codec_config).to(device)
        layer = conv_gaines(weight_cpu.to(device), None, stride=1, padding=1,
                            config=conv_config).to(device)
        spikes = [encoder(input_cpu.to(device)).clone() for _ in range(timestep)]
        encoder.reset()

        first = torch.stack([layer(spike).clone() for spike in spikes])
        assert layer.timestep_cur == timestep
        layer.reset()
        assert layer.timestep_cur == 0
        second = torch.stack([layer(spike).clone() for spike in spikes])
        assert torch.equal(first, second), f'[{device}] replay after reset diverged'
        print(f'[{device}] reset and replay reproduce {tuple(first.shape)} spikes.')


def _performance_check():
    """Time conv_gaines against conv_mix on identical spikes."""
    ntype = global_config.ntype
    timestep = 256
    torch.manual_seed(42)  # Keep gen_rand_tensor inputs reproducible.
    input_cpu = gen_rand_tensor('bipolar', (2, 2, 8, 8), 8).type(ntype)
    weight_cpu = gen_rand_tensor('bipolar', (4, 2, 2, 2), 8).type(ntype)
    codec_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
    for device in devices():
        input_x = input_cpu.to(device)
        weight = weight_cpu.to(device)
        encoder = encode(codec_config).to(device)
        gaines = conv_gaines(weight, None, stride=1, padding=0,
                             config={'polarity': 'bipolar', 'timestep': timestep,
                                     'generator': 'sobol', 'dim': 2, 'scaled': True}).to(device)
        mixed = conv_mix(weight, None, stride=1, padding=0,
                         config={'polarity': 'bipolar', 'timestep': timestep,
                                 'generator': 'sobol', 'dim': 20, 'width': 12}).to(device)

        spikes = [encoder(input_x).clone() for _ in range(timestep)]
        encoder.reset()
        with timer(device) as t_gaines:
            for spike in spikes:
                gaines(spike)
        gaines.reset()
        with timer(device) as t_mixed:
            for spike in spikes:
                mixed(spike)
        mixed.reset()
        gaines_seconds = t_gaines.seconds or 0.0
        mixed_seconds = t_mixed.seconds or 0.0
        print(f'[{device}] perf: conv_gaines {gaines_seconds*1e3:.1f}ms vs '
              f'conv_mix {mixed_seconds*1e3:.1f}ms '
              f'(ratio {mixed_seconds/max(gaines_seconds, 1e-9):.2f}x)')


def _kernel_specific_checks():
    _fidelity_checks()
    _geometry_equivalence_check()
    _pad_decorrelation_check()
    _rejection_checks()
    _known_answer_check()
    _non_scaled_check()
    _reset_replay_check()
    _performance_check()
    print('Test passed.')


# The scaled Gaines MUX cycles through the fan-in on a period-entry schedule, so a
# small entry lands on a biased Sobol sub-lattice; 16 keeps the fidelity error low.
_SUITE_CHANNELS = 16


def _suite_weight(polarity):
    low, high = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return torch.linspace(low, high, _SUITE_CHANNELS,
                          dtype=global_config.ntype).reshape(1, _SUITE_CHANNELS, 1, 1)


def make_operation(polarity, timestep, _device):
    return conv_gaines(
        _suite_weight(polarity), None, stride=1, padding=0,
        config={'polarity': polarity, 'timestep': timestep,
                'generator': 'sobol', 'dim': 2, 'scaled': True},
    )


def make_values(polarity):
    low, high = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(low, high, _SUITE_CHANNELS * 16)
                 .reshape(1, _SUITE_CHANNELS, 4, 4),)


def make_random_perf_values(polarity):
    return (make_values(polarity)[0].repeat(512, 1, 1, 1),)


def analytic_reference(values, polarity):
    return _scaled_reference(values[0], _suite_weight(polarity), None, 0,
                             _SUITE_CHANNELS, polarity)


def known_answer_case(polarity):
    values = torch.ones(1, _SUITE_CHANNELS, 2, 2)
    return (
        (values,),
        analytic_reference((values,), polarity),
    )


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    # 16 kernel positions need a sequence period above 16 to stay distinct.
    'state_timesteps': 32,
    'extra_checks': _kernel_specific_checks,
}


def test_conv_gaines():
    """Verify conv_gaines for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_conv_gaines()

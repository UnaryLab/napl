import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.module import linear_gaines2, linear
from napl.sim.operation import encode, decode


class napl_linear_gaines2(napl_base):
    """Canonical wiring: encoder -> linear_gaines2 -> decoder."""


    def __init__(self, codec_config, layer_config, weight, bias):
        super().__init__()
        self.encoder = encode(codec_config)
        self.layer = linear_gaines2(weight, bias, layer_config)
        self.decoder = decode({'polarity': layer_config['polarity'],
                                'timestep': codec_config['timestep']})


    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        i_spike = self.encoder(input_x)
        o_spike = self.layer(i_spike)
        self.decoder(o_spike)


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.encoder.reset()
        self.layer.reset()
        self.decoder.reset()


def _kernel_specific_checks():
    """
    The streaming Gaines linear (linear_gaines2) reproduces (W x + b - 1) / entry in scaled
    mode within a stochastic-computing bound, for both polarities, on every device; in
    non-scaled mode the bipolar saturating counter tracks the sign of the sum and the
    unipolar OR emits no spike on a zero count. UnarySim reference: GainesLinear4.
    """
    timestep = 2048

    out_features = 8

    # A power-of-two entry count makes the Gaines threshold scale exact. The
    # duplicated threshold state and clipped count add an O(1/L) offset, so bounds
    # target entry=16 rather than 1/sqrt(T). LFSR input avoids phase-locking with
    # the periodic threshold table.
    for device in devices():
        for polarity in ['unipolar', 'bipolar']:
            for has_bias in [True, False]:
                in_features = 15 if has_bias else 16
                entry = in_features + (1 if has_bias else 0)
                input_x = gen_rand_tensor(polarity, shape=(in_features,), width=8).type(global_config.ntype).to(device)
                weight = gen_rand_tensor(polarity, shape=(out_features, in_features), width=8).type(global_config.ntype).to(device)
                bias = gen_rand_tensor(polarity, shape=(out_features,), width=8).type(global_config.ntype).to(device) if has_bias else None

                codec_config = {'polarity': polarity, 'timestep': timestep, 'generator': 'lfsr', 'seed': 999}
                layer_config = {'polarity': polarity, 'timestep': timestep, 'generator': 'lfsr',
                                'dim': 2, 'seed': 1, 'scaled': True}

                inst = napl_linear_gaines2(codec_config, layer_config, weight, bias).to(device)
                inst(input_x, timesteps=timestep)

                # gt(c, v) equals ge(c - 1, v) for the integer count and levels, so the
                # strict scaled adder recovers one count below the plain scaling.
                y = weight @ input_x + (bias if has_bias else 0) - 1
                L = entry  # The power-of-two entry count equals scale_len.
                if polarity == 'bipolar':
                    ref = (entry + y) / (L - 1) - 1
                else:
                    ref = y / (L - 1)
                val = inst.decoder.spike_value
                err = (val - ref).abs()
                rmse = torch.sqrt(err.pow(2).mean()).item()
                assert err.max().item() < 2.0 / L + 3.0 / (timestep ** 0.5), \
                    f'{device}/{polarity}/bias={has_bias}: max err {err.max().item()} too large'
                assert rmse < 1.5 / L + 3.0 / (timestep ** 0.5), \
                    f'{device}/{polarity}/bias={has_bias}: rmse {rmse} exceeds Gaines bound'
                assert inst.layer.timestep_cur == timestep
                print(f'[{device}] scaled {polarity} bias={has_bias}: rmse={rmse:.5f} '
                      f'max_err={err.max().item():.5f}')
                inst.reset()

    in_features = 16

    # The non-scaled bipolar counter rails to sign(Wx); these rows stay away from zero.
    w_sign_cpu = torch.cat([torch.full((out_features // 2, in_features), 0.5),
                            torch.full((out_features - out_features // 2, in_features), -0.5)]
                           ).type(global_config.ntype)
    x_half_cpu = torch.full((in_features,), 0.5).type(global_config.ntype)
    w1_cpu = gen_rand_tensor('unipolar', shape=(out_features, in_features), width=8).type(global_config.ntype)
    x0_cpu = torch.zeros(in_features).type(global_config.ntype)
    for device in devices():
        w_sign = w_sign_cpu.to(device)
        x_half = x_half_cpu.to(device)
        inst = napl_linear_gaines2(
            {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1},
            {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'lfsr', 'dim': 2, 'seed': 1,
             'scaled': False, 'depth': 8},
            w_sign, None).to(device)
        inst(x_half, timesteps=timestep)
        val = inst.decoder.spike_value
        assert (val[:out_features // 2] > 0.8).all(), f'[{device}] non-scaled bipolar positive rail: {val}'
        assert (val[out_features // 2:] < -0.8).all(), f'[{device}] non-scaled bipolar negative rail: {val}'
        print(f'[{device}] non-scaled bipolar sign rails: {val.tolist()}')
        inst.reset()

        w1 = w1_cpu.to(device)
        x0 = x0_cpu.to(device)
        inst = napl_linear_gaines2(
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 1},
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'lfsr', 'dim': 2, 'seed': 1,
             'scaled': False},
            w1, None).to(device)
        inst(x0, timesteps=64)
        assert torch.equal(inst.decoder.spike_count, torch.zeros_like(inst.decoder.spike_count)), \
            f'[{device}] non-scaled unipolar all-zero input must emit no spikes'
        inst.reset()
    print('non-scaled unipolar zero-input corner passed.')

    # The weight path inlines encode's comparison instead of composing an encode
    # instance, because each input feature carries its own sequence. This pins
    # that equivalence over a full period so the inline form cannot drift.
    for polarity in ['unipolar', 'bipolar']:
        period = 64
        lo = 0.0 if polarity == 'unipolar' else -1.0
        w_ref = gen_rand_tensor(polarity, shape=(out_features, in_features), width=8).type(global_config.ntype)
        layer = linear_gaines2(w_ref, None, {'polarity': polarity, 'timestep': period,
                                             'generator': 'lfsr', 'dim': 2, 'seed': 1,
                                             'scaled': True})
        column = 3
        ref_enc = encode({'polarity': polarity, 'timestep': period, 'generator': 'lfsr',
                          'dim': 2 + column, 'seed': 1 + column})
        for t in range(layer.w_len):
            w_prob = (layer.weight + 1) / 2 if polarity == 'bipolar' else layer.weight
            inline = torch.gt(w_prob.t(), layer.w_num_seq[t].unsqueeze(-1))[column]
            assert torch.equal(inline.type(global_config.stype), ref_enc(w_ref[:, column])), \
                f'{polarity} t={t}: inline weight encoding diverged from a composed encode instance'
    print('inline weight path matches a composed encode over a full period, both polarities.')

    # The weight probability is cached across timesteps, so a stale cache would
    # silently keep encoding the old weights or a tensor on the wrong device.
    period = 64
    # The layer wraps the given tensor in a Parameter without copying, so each
    # instance below gets its own clone.
    w_lo = torch.full((out_features, in_features), -0.9).type(global_config.ntype)
    layer = linear_gaines2(w_lo.clone(), None, {'polarity': 'bipolar', 'timestep': period,
                                                'generator': 'lfsr', 'dim': 2, 'seed': 1,
                                                'scaled': True})
    x_ones = torch.ones(in_features).type(global_config.ntype)
    before = torch.stack([layer(x_ones).clone() for _ in range(period)])
    layer.reset()
    with torch.no_grad():
        layer.weight.fill_(0.9)
    after = torch.stack([layer(x_ones).clone() for _ in range(period)])
    assert not torch.equal(before, after), \
        'weight probability cache went stale after an in-place weight update'
    for device in devices():
        moved = linear_gaines2(w_lo.clone(), None, {'polarity': 'bipolar', 'timestep': period,
                                                    'generator': 'lfsr', 'dim': 2, 'seed': 1,
                                                    'scaled': True})
        moved(torch.ones(in_features).type(global_config.ntype))  # Prime the cache on CPU.
        moved = moved.to(device)
        moved.reset()
        out = torch.stack([moved(x_ones.to(device)).clone() for _ in range(period)])
        assert out.device.type == torch.empty(0, device=device).device.type, \
            f'{device}: output left the target device'
        assert torch.equal(out.cpu(), before), \
            f'{device}: cached weight probability did not follow the module across devices'
    print('weight probability cache invalidates on weight update and on device migration.')

    # Seeds are derived per input feature and for the scaled threshold, so they run
    # past the LFSR period on a wide layer and land on a multiple of it. Both sites
    # once failed at construction: the per-column seeds when in_features reaches
    # 2**width, and the scaled-threshold seed at the widths below.
    for period, features in [(256, 256), (256, 512), (64, 64)]:
        layer = linear_gaines2(torch.zeros(4, features).type(global_config.ntype), None,
                               {'polarity': 'bipolar', 'timestep': period, 'generator': 'lfsr',
                                'dim': 2, 'seed': 1, 'scaled': True})
        assert layer.w_num_seq.shape == (layer.w_len, features)
        assert layer(torch.ones(features).type(global_config.ntype)).shape == (4,)
    for features in [6, 14, 30]:
        linear_gaines2(torch.zeros(4, features).type(global_config.ntype), None,
                       {'polarity': 'bipolar', 'timestep': 256, 'generator': 'lfsr',
                        'dim': 2, 'seed': 1, 'scaled': True})
    print('lfsr seeds that reduce to zero construct at both the per-column and scale sites.')

    # An lfsr seed reducing to zero aliases onto the all-ones seed, so a layer wider
    # than the sequence period silently gives two features one sequence. The layer
    # must report that rather than let the comonotone weight bits pass unseen.
    wide = linear_gaines2(torch.zeros(4, 256).type(global_config.ntype), None,
                          {'polarity': 'bipolar', 'timestep': 256, 'generator': 'lfsr',
                           'dim': 2, 'seed': 1, 'scaled': True})
    assert torch.unique(wide.w_num_seq, dim=1).shape[1] < wide.in_features, \
        'expected an aliased column pair at in_features equal to the sequence period'
    narrow = linear_gaines2(torch.zeros(4, 32).type(global_config.ntype), None,
                            {'polarity': 'bipolar', 'timestep': 256, 'generator': 'lfsr',
                             'dim': 2, 'seed': 1, 'scaled': True})
    assert torch.unique(narrow.w_num_seq, dim=1).shape[1] == narrow.in_features, \
        'well-sized layer must derive one distinct sequence per input feature'
    print('duplicate weight sequences are detected when in_features reaches the period.')

    # The suite drives this layer with an explicit lfsr generator, so the default
    # path needs its own check. Sobol draws every level in 0..L-1 once, making the
    # scaled adder's count-to-rate map exactly count / L with no offset.
    # Constructed with no config at all, since required keys are not merged
    # per-key: the signature default applies only when config is omitted.
    default_layer = linear_gaines2(torch.zeros(4, 16).type(global_config.ntype))
    assert default_layer.reference_encode.generator == 'sobol', \
        'linear_gaines2 must default to the sobol generator'
    levels = sorted(int(v) for v in default_layer.scale_seq.tolist())
    L = default_layer.scale_len
    assert levels == list(range(L)), f'sobol scale_seq must cover 0..{L - 1} once, got {levels}'
    for count in range(L + 1):
        rate = sum(1 for t in range(L) if count / L > default_layer.scale_seq[t] / L) / L
        assert abs(rate - count / L) < 1e-12, \
            f'default sobol transfer must be count/L exactly; count {count} gave {rate}'
    print('default generator is sobol and its scaled transfer is exactly count/L.')

    # Compare linear_gaines2 and linear on identical spikes.
    input_x_cpu = gen_rand_tensor('bipolar', shape=(in_features,), width=8).type(global_config.ntype)
    weight_cpu = gen_rand_tensor('bipolar', shape=(out_features, in_features), width=8).type(global_config.ntype)
    for device in devices():
        input_x = input_x_cpu.to(device)
        weight = weight_cpu.to(device)
        enc = encode({'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}).to(device)
        gaines = linear_gaines2(weight, None, {'polarity': 'bipolar', 'timestep': timestep,
                                               'generator': 'lfsr', 'dim': 2, 'seed': 1,
                                               'scaled': True}).to(device)
        lin = linear(weight, None, {'polarity': 'bipolar', 'timestep': timestep,
                                        'generator': 'sobol', 'dim': 2, 'scale': None, 'width': 12}).to(device)

        spikes = [enc(input_x).clone() for _ in range(timestep)]
        enc.reset()
        with timer(device) as t_g:
            for s in spikes:
                gaines(s)
        gaines.reset()
        with timer(device) as t_f:
            for s in spikes:
                lin(s)
        lin.reset()
        print(f'[{device}] perf: linear_gaines2 {t_g.seconds*1e3:.1f}ms vs linear {t_f.seconds*1e3:.1f}ms '
              f'(ratio {t_f.seconds/max(t_g.seconds,1e-9):.2f}x)')

    print('Test passed.')


def _suite_weight(polarity):
    if polarity == 'unipolar':
        return torch.tensor([
            [0.25, 0.5, 0.75, 1.0],
            [1.0, 0.75, 0.5, 0.25],
        ])
    return torch.tensor([
        [-0.75, -0.25, 0.25, 0.75],
        [0.75, 0.25, -0.25, -0.75],
    ])


def make_operation(polarity, timestep, _device):
    return linear_gaines2(
        _suite_weight(polarity), None,
        {
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'lfsr',
            'dim': 2,
            'seed': 1,
            'scaled': True,
        },
    )


def make_values(polarity):
    low, high = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(low, high, 4),)


def make_performance_values(polarity):
    values = make_values(polarity)[0]
    return (values.repeat(32768, 1),)


def analytic_reference(values, polarity):
    # The scaled adder compares strictly, and gt(c, v) equals ge(c - 1, v) for the
    # integer count c and integer levels, so the recovered value sits one count
    # below the plain count/(L-1) map.
    result = _suite_weight(polarity) @ values[0] - 1
    if polarity == 'bipolar':
        return (4 + result) / 3 - 1
    return result / 3


def known_answer_case(polarity):
    low, high = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    values = torch.linspace(low, high, 4)
    return (
        (values,),
        analytic_reference((values,), polarity),
        5.0 / (256 ** 0.5),
    )


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 5.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_generators': ['lfsr'],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_linear_gaines2():
    """Verify linear_gaines2 against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_linear_gaines2()

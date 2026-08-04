import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.module.encoder import encoder
from napl.sim.module.decoder import decoder
from napl.sim.module.linear_gaines4 import linear_gaines4
from napl.sim.module.linear import linear


class napl_linear_gaines4(napl_base):
    """Canonical wiring: encoder -> linear_gaines4 -> decoder."""


    def __init__(self, codec_config, layer_config, weight, bias):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.layer = linear_gaines4(weight, bias, layer_config)
        self.decoder = decoder({'polarity': layer_config['polarity'],
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
    The streaming Gaines linear (linear_gaines4) reproduces (W x + b) / entry in scaled
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

                inst = napl_linear_gaines4(codec_config, layer_config, weight, bias).to(device)
                inst(input_x, timesteps=timestep)

                y = weight @ input_x + (bias if has_bias else 0)
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
        inst = napl_linear_gaines4(
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
        inst = napl_linear_gaines4(
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 1},
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'lfsr', 'dim': 2, 'seed': 1,
             'scaled': False},
            w1, None).to(device)
        inst(x0, timesteps=64)
        assert torch.equal(inst.decoder.spike_count, torch.zeros_like(inst.decoder.spike_count)), \
            f'[{device}] non-scaled unipolar all-zero input must emit no spikes'
        inst.reset()
    print('non-scaled unipolar zero-input corner passed.')

    # Compare linear_gaines4 and linear on identical spikes.
    input_x_cpu = gen_rand_tensor('bipolar', shape=(in_features,), width=8).type(global_config.ntype)
    weight_cpu = gen_rand_tensor('bipolar', shape=(out_features, in_features), width=8).type(global_config.ntype)
    for device in devices():
        input_x = input_x_cpu.to(device)
        weight = weight_cpu.to(device)
        enc = encoder({'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}).to(device)
        gaines = linear_gaines4(weight, None, {'polarity': 'bipolar', 'timestep': timestep,
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
        print(f'[{device}] perf: linear_gaines4 {t_g.seconds*1e3:.1f}ms vs linear {t_f.seconds*1e3:.1f}ms '
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
    return linear_gaines4(
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
    result = _suite_weight(polarity) @ values[0]
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


def test_linear_gaines4():
    """Verify linear_gaines4 against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_linear_gaines4()

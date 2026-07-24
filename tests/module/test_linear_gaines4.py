import time
import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module.encoder import encoder
from napl.sim.module.decoder import decoder
# import directly from the module (not the package): the class is not wired into
# module/__init__.py yet.
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


def test_linear_gaines4():
    """
    The streaming Gaines linear (linear_gaines4) reproduces (W x + b) / entry in scaled
    mode within a stochastic-computing bound, for both polarities, on every device; in
    non-scaled mode the bipolar saturating counter tracks the sign of the sum and the
    unipolar OR emits no spike on a zero count. UnarySim reference: GainesLinear4.
    """
    timestep = 2048

    out_features = 8

    # scaled mode: entry a power of two so the Gaines threshold scale is exact.
    # Reference is the idealized Gaines scaled add y/(L-1) (lfsr thresholds {1..L-1},
    # compare >=). The real mechanism carries an O(1/L) systematic offset on top of the
    # SC noise (threshold table has a duplicated state, count clips at L-1; identical
    # quirks in UnarySim GainesLinear4), so the bounds are sized for entry = 16, not
    # 1/sqrt(T). The input codec is lfsr (the Gaines flavor): a sobol dim-1 input
    # phase-locks with the L-periodic threshold table and biases the result.
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
                L = entry  # entry is a power of two here, so scale_len == entry
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

    # non-scaled bipolar: the saturating counter rails to sign(W x); +/-0.5 everywhere
    # gives a clearly positive (first half) / negative (second half) sum per output row
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

    # performance: time linear_gaines4 vs linear on identical spike streams
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
        sync(device)
        t0 = time.time()
        for s in spikes:
            gaines(s)
        sync(device)
        t_g = time.time() - t0
        gaines.reset()
        sync(device)
        t0 = time.time()
        for s in spikes:
            lin(s)
        sync(device)
        t_f = time.time() - t0
        lin.reset()
        print(f'[{device}] perf: linear_gaines4 {t_g*1e3:.1f}ms vs linear {t_f*1e3:.1f}ms '
              f'(ratio {t_f/max(t_g,1e-9):.2f}x)')

    print('Test passed.')


if __name__ == '__main__':
    test_linear_gaines4()

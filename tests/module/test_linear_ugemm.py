import time
import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, sync
from napl.sim.module import encoder, decoder
from napl.sim.module.linear import linear
from napl.sim.module.linear_ugemm import linear_ugemm


class napl_linear_ugemm(napl_base):
    """Wire encoder -> linear_ugemm -> decoder (canonical streaming round-trip)."""


    def __init__(self, codec_config, lin_config, weight, bias):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.linear = linear_ugemm(weight, bias, lin_config)


    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        i_spike = self.encoder(input_x)
        o_spike = self.linear(i_spike)
        self.decoder(o_spike)


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.encoder.reset()
        self.decoder.reset()
        self.linear.reset()


def _kernel_specific_checks():
    """
    The streaming uGEMM linear layer (conditional-spike-generated weight bits + scaled
    unary adder) reproduces (W x + b) / entry within the stochastic-computing bound, for
    both polarities, on every device. UnarySim reference: FSULinearuGEMM (scaled=True).
    """
    timestep = 256
    in_features, out_features = 16, 8

    for device in devices():
        for polarity in ['unipolar', 'bipolar']:
            for has_bias in [True, False]:
                input_x = gen_rand_tensor(polarity, shape=(in_features,), width=8).type(global_config.ntype).to(device)
                weight = gen_rand_tensor(polarity, shape=(out_features, in_features), width=8).type(global_config.ntype).to(device)
                bias = gen_rand_tensor(polarity, shape=(out_features,), width=8).type(global_config.ntype).to(device) if has_bias else None

                codec_config = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 2}
                lin_config = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 1,
                              'scale': None, 'width': 12}

                inst = napl_linear_ugemm(codec_config, lin_config, weight, bias).to(device)
                inst(input_x, timesteps=timestep)

                entry = in_features + (1 if has_bias else 0)
                ref = (weight @ input_x + (bias if has_bias else 0)) / entry
                err = (inst.decoder.spike_value - ref).abs()
                rmse = torch.sqrt(err.pow(2).mean()).item()
                assert err.max().item() < 0.05, \
                    f'{device}/{polarity}/bias={has_bias}: max err {err.max().item()} too large'
                assert inst.linear.timestep_cur == timestep
                print(f'[{device}] {polarity} bias={has_bias}: rmse={rmse:.5f} max_err={err.max().item():.5f}')
                inst.reset()

    # All-ones unipolar operands make the scaled adder emit 1 every timestep.
    w1_cpu = torch.ones(out_features, in_features).type(global_config.ntype)
    x1_cpu = torch.ones(in_features).type(global_config.ntype)
    cfg = {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 1}
    for device in devices():
        w1 = w1_cpu.to(device)
        x1 = x1_cpu.to(device)
        inst = napl_linear_ugemm(cfg, {**cfg, 'scale': None, 'width': 12}, w1, None).to(device)
        inst(x1, timesteps=64)
        assert torch.allclose(inst.decoder.spike_value, torch.ones(out_features, device=device)), \
            f'[{device}] all-ones unipolar uGEMM linear should decode to exactly 1'
        inst.reset()
    print('known-answer corner passed.')

    # Compare linear_ugemm and linear on identical spikes.
    input_x_cpu = gen_rand_tensor('bipolar', shape=(in_features,), width=8).type(global_config.ntype)
    weight_cpu = gen_rand_tensor('bipolar', shape=(out_features, in_features), width=8).type(global_config.ntype)
    bias_cpu = gen_rand_tensor('bipolar', shape=(out_features,), width=8).type(global_config.ntype)
    lin_cfg = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1,
               'scale': None, 'width': 12}
    for device in devices():
        input_x = input_x_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        enc = encoder({'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2}).to(device)
        ug = linear_ugemm(weight, bias, lin_cfg).to(device)
        lin = linear(weight, bias, {**lin_cfg, 'dim': 3}).to(device)

        spikes = [enc(input_x).clone() for _ in range(timestep)]
        enc.reset()
        sync(device)
        t0 = time.time()
        for s in spikes:
            ug(s)
        sync(device)
        t_ug = time.time() - t0
        ug.reset()
        sync(device)
        t0 = time.time()
        for s in spikes:
            lin(s)
        sync(device)
        t_lin = time.time() - t0
        lin.reset()
        print(f'[{device}] perf: linear_ugemm {t_ug*1e3:.1f}ms vs linear {t_lin*1e3:.1f}ms '
              f'(ratio {t_lin/max(t_ug,1e-9):.2f}x)')

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
    return linear_ugemm(
        _suite_weight(polarity), None,
        {
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 1,
            'scale': None,
            'width': 12,
        },
    )


def make_values(polarity):
    if polarity == 'unipolar':
        return (torch.tensor([0.1, 0.3, 0.5, 0.7]),)
    return (torch.tensor([-0.75, -0.25, 0.25, 0.75]),)


def analytic_reference(values, polarity):
    return _suite_weight(polarity) @ values[0] / 4


def known_answer_case(polarity):
    values = torch.ones(4)
    return (
        (values,),
        _suite_weight(polarity) @ values / 4,
        3.0 / (256 ** 0.5),
    )


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 3.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [2],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_linear_ugemm():
    """Verify linear_ugemm against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_linear_ugemm()

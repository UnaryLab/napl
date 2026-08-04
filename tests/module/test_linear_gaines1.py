import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.module import encoder, decoder
from napl.sim.module.linear import linear
# This class is not exported from module/__init__.
from napl.sim.module.linear_gaines1 import linear_gaines1


class napl_linear_gaines1(napl_base):
    """Wire encoder -> linear_gaines1 -> decoder (canonical round-trip)."""


    def __init__(self, codec_config, lin_config, weight, bias):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.linear = linear_gaines1(weight, bias, lin_config)


    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        i_spike = self.encoder(input_x)
        o_spike = self.linear(i_spike)
        self.decoder(o_spike)


def _scaled_ref(s, polarity, entry):
    """Decoded value the Gaines scaled adder converges to for inner product s."""
    w = 2 ** round(torch.log2(torch.tensor(float(entry))).item())
    if polarity == 'bipolar':
        return ((entry + s) / w - 1).clamp(-1, 1)
    return (s / w).clamp(0, 1)


def _kernel_specific_checks():
    """
    Streaming Gaines linear (gMUL + gADD) reproduces its analytic target within a
    stochastic-computing bound on every device: scaled mode tracks the scaled inner
    product, non-scaled bipolar tracks clamp(W x + b, -1, 1). UnarySim: GainesLinear1.
    """
    timestep = 1024
    in_features, out_features = 16, 8
    torch.manual_seed(42)  # Keep gen_rand_tensor inputs reproducible.

    for device in devices():
        # Cover both polarities and bias settings in scaled mode.
        for polarity in ['unipolar', 'bipolar']:
            for has_bias in [True, False]:
                input_x = gen_rand_tensor(polarity, shape=(in_features,), width=8).type(global_config.ntype).to(device)
                weight = gen_rand_tensor(polarity, shape=(out_features, in_features), width=8).type(global_config.ntype).to(device)
                bias = gen_rand_tensor(polarity, shape=(out_features,), width=8).type(global_config.ntype).to(device) if has_bias else None

                codec_config = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
                lin_config = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 2, 'scaled': True}

                inst = napl_linear_gaines1(codec_config, lin_config, weight, bias).to(device)
                inst(input_x, timesteps=timestep)

                entry = in_features + (1 if has_bias else 0)
                s = weight @ input_x + (bias if has_bias else 0)
                r_value = _scaled_ref(s, polarity, entry)

                err = (inst.decoder.spike_value - r_value).abs()
                rmse = torch.sqrt(err.pow(2).mean()).item()
                print(f'[{device}] scaled {polarity} bias={has_bias}: rmse={rmse:.5f} max_err={err.max().item():.5f}')
                assert rmse < 0.1, f'{device}/scaled/{polarity}/bias={has_bias}: rmse {rmse}'
                assert inst.linear.timestep_cur == timestep
                inst.reset()

        # The non-scaled bipolar counter settles at sign(Wx+b) away from zero.
        # Near zero it random-walks without a well-defined steady state.
        input_x = gen_rand_tensor('bipolar', shape=(in_features,), width=8).type(global_config.ntype).to(device)
        weight = gen_rand_tensor('bipolar', shape=(out_features, in_features), width=8).type(global_config.ntype).to(device)
        codec_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
        lin_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2, 'scaled': False, 'depth': 8}
        inst = napl_linear_gaines1(codec_config, lin_config, weight, None).to(device)
        inst(input_x, timesteps=timestep)
        s = weight @ input_x
        mask = s.abs() > 1
        assert mask.any(), 'seed produced no |s|>1 element; adjust seed'
        err = (inst.decoder.spike_value - torch.sign(s))[mask].abs()
        print(f'[{device}] non-scaled bipolar (|s|>1, n={int(mask.sum())}): max_err={err.max().item():.5f}')
        assert err.max().item() < 0.15, f'{device}/non-scaled/bipolar: max err {err.max().item()}'
        inst.reset()

    # All-ones unipolar operands make the scaled output emit 1 every timestep.
    w1_cpu = torch.ones(out_features, in_features).type(global_config.ntype)
    x1_cpu = torch.ones(in_features).type(global_config.ntype)
    for device in devices():
        w1 = w1_cpu.to(device)
        x1 = x1_cpu.to(device)
        inst = napl_linear_gaines1(
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 1},
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 2, 'scaled': True},
            w1, None).to(device)
        inst(x1, timesteps=64)
        assert torch.allclose(inst.decoder.spike_value, torch.ones(out_features, device=device)), \
            f'[{device}] all-ones unipolar scaled Gaines linear should spike every timestep'
        inst.reset()
    print('known-answer corner passed.')

    # Compare the comparator and scaled accumulator adders on identical spikes.
    input_x_cpu = gen_rand_tensor('bipolar', shape=(in_features,), width=8).type(global_config.ntype)
    weight_cpu = gen_rand_tensor('bipolar', shape=(out_features, in_features), width=8).type(global_config.ntype)
    bias_cpu = gen_rand_tensor('bipolar', shape=(out_features,), width=8).type(global_config.ntype)
    cfg = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2}
    for device in devices():
        input_x = input_x_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        enc = encoder({'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}).to(device)
        gl = linear_gaines1(weight, bias, {**cfg, 'scaled': True}).to(device)
        lin = linear(weight, bias, {**cfg, 'scale': None, 'width': 12}).to(device)

        spikes = [enc(input_x).clone() for _ in range(timestep)]
        enc.reset()
        with timer(device) as t_gl:
            for s in spikes:
                gl(s)
        gl.reset()
        with timer(device) as t_lin:
            for s in spikes:
                lin(s)
        lin.reset()
        print(f'[{device}] perf: linear_gaines1 {t_gl.seconds*1e3:.1f}ms vs linear {t_lin.seconds*1e3:.1f}ms '
              f'(ratio {t_lin.seconds/max(t_gl.seconds,1e-9):.2f}x)')

    print('Test passed.')


def _suite_weight(polarity):
    # The 16-entry suite uses a 4-bit threshold RNG.
    if polarity == 'unipolar':
        row = torch.linspace(0.0, 1.0, 16)
    else:
        row = torch.linspace(-1.0, 1.0, 16)
    return torch.stack((row, row.flip(0)))


def make_operation(polarity, timestep, _device):
    return linear_gaines1(
        _suite_weight(polarity), None,
        {
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 2,
            'scaled': True,
        },
    )


def make_values(polarity):
    low, high = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(low, high, 16),)


def make_performance_values(polarity):
    values = make_values(polarity)[0]
    return (values.repeat(32768, 1),)


def analytic_reference(values, polarity):
    return _scaled_ref(
        _suite_weight(polarity) @ values[0], polarity, 16
    )


def known_answer_case(polarity):
    values = torch.ones(16)
    return (
        (values,),
        analytic_reference((values,), polarity),
        3.2 / (256 ** 0.5),
    )


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 3.2,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_linear_gaines1():
    """Verify linear_gaines1 against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_linear_gaines1()

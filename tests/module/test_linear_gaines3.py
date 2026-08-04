import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.module.encoder import encoder
from napl.sim.module.linear_gaines3 import linear_gaines3
from napl.sim.module.linear import linear


class napl_linear_gaines3(napl_base):
    """Wire encoder -> linear_gaines3 and count the output spikes."""


    def __init__(self, codec_config, lin_config, weight, bias):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.lin = linear_gaines3(weight, bias, lin_config)
        self.acc = None


    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        i_spike = self.encoder(input_x)
        o_spike = self.lin(i_spike)
        count = o_spike.type(global_config.ntype)
        self.acc = count if self.acc is None else self.acc + count


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.encoder.reset()
        self.lin.reset()
        self.acc = None


def _kernel_specific_checks():
    """
    The streaming Gaines linear (uMUL + gADD, scaled) recovers (W x + b) / 2**round(log2(entry))
    within the stochastic-computing bound on every device, both polarities, with/without bias.
    Note the shared weight RNG (faithful to GainesLinear3) correlates the weight streams, and
    the ge-comparison scaled adder carries a +1/scale offset, so the bound is looser than the streaming linear.
    """
    timestep = 256
    out_features = 8
    # A power-of-two entry count makes the gADD scale exact.
    for has_bias in [False, True]:
        in_features = 64 - (1 if has_bias else 0)
        entry = 64
        for polarity in ['unipolar', 'bipolar']:
            for device in devices():
                input_x = gen_rand_tensor(polarity, shape=(in_features,), width=8).type(global_config.ntype).to(device)
                weight = gen_rand_tensor(polarity, shape=(out_features, in_features), width=8).type(global_config.ntype).to(device)
                bias = gen_rand_tensor(polarity, shape=(out_features,), width=8).type(global_config.ntype).to(device) if has_bias else None

                inst = napl_linear_gaines3(
                    {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 2},
                    {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol'},
                    weight, bias).to(device)
                inst(input_x, timesteps=timestep)

                mean = inst.acc / timestep
                value = 2 * mean - 1 if polarity == 'bipolar' else mean
                ref = (torch.matmul(weight, input_x) + (bias if has_bias else 0)) / entry
                err = (value - ref).abs()
                rmse = torch.sqrt(err.pow(2).mean()).item()
                assert rmse < 0.1, f'{device}/{polarity}/bias={has_bias}: rmse {rmse}'
                assert err.max().item() < 0.2, f'{device}/{polarity}/bias={has_bias}: max err {err.max().item()}'
                assert inst.lin.timestep_cur == timestep
                print(f'{device}/{polarity}/bias={has_bias}: rmse={rmse:.5f} max_err={err.max().item():.5f}')
                inst.reset()

    # Integer spike streams must produce bit-exact counts across devices.
    if len(devices()) > 1:
        accs = []
        for device in devices():
            torch.manual_seed(0)
            weight = gen_rand_tensor('bipolar', shape=(out_features, 64), width=8).type(global_config.ntype).to(device)
            input_x = gen_rand_tensor('bipolar', shape=(64,), width=8).type(global_config.ntype).to(device)
            inst = napl_linear_gaines3(
                {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2},
                {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol'},
                weight, None).to(device)
            inst(input_x, timesteps=timestep)
            accs.append(inst.acc.cpu())
        for a in accs[1:]:
            assert torch.equal(accs[0], a), 'cross-device spike counts diverge'
        print('cross-device bit-exactness passed.')

    n = 16
    w1_cpu = torch.ones(out_features, n).type(global_config.ntype)
    x1_cpu = torch.ones(n).type(global_config.ntype)
    x0_cpu = torch.zeros(n).type(global_config.ntype)
    for device in devices():
        w1 = w1_cpu.to(device)
        x1 = x1_cpu.to(device)
        x0 = x0_cpu.to(device)
        inst = napl_linear_gaines3(
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 2},
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol'},
            w1, None).to(device)
        inst(x1, timesteps=64)
        assert torch.equal(inst.acc, torch.full((out_features,), 64.0, device=device)), \
            f'[{device}] all-ones scaled unipolar must be constant 1'
        for x, expect in [(x0, 0.0), (x1, 64.0)]:
            inst = napl_linear_gaines3(
                {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 2},
                {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'scaled': False},
                w1, None).to(device)
            inst(x, timesteps=64)
            assert torch.equal(inst.acc, torch.full((out_features,), expect, device=device)), \
                f'[{device}] non-scaled unipolar OR corner (expect {expect})'
        inst = napl_linear_gaines3(
            {'polarity': 'bipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 2},
            {'polarity': 'bipolar', 'timestep': 64, 'generator': 'sobol', 'scaled': False},
            w1, None).to(device)
        inst(x1, timesteps=64)
        assert (2 * inst.acc / 64 - 1).min().item() > 0.9, \
            f'[{device}] non-scaled bipolar saturation corner'
    print('known-answer corners passed.')

    # Compare linear_gaines3 and linear on identical inputs per device.
    in_features = 64
    input_x_cpu = gen_rand_tensor('bipolar', shape=(in_features,), width=8).type(global_config.ntype)
    weight_cpu = gen_rand_tensor('bipolar', shape=(out_features, in_features), width=8).type(global_config.ntype)
    for device in devices():
        input_x = input_x_cpu.to(device)
        weight = weight_cpu.to(device)
        enc = encoder({'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}).to(device)
        gl = linear_gaines3(weight, None, {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol'}).to(device)
        lin = linear(weight, None, {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2, 'width': 12}).to(device)
        times = {}
        for name, mod in [('gaines3', gl), ('lin', lin)]:
            enc.reset()
            mod.reset()
            with timer(device) as elapsed:
                for _ in range(timestep):
                    mod(enc(input_x))
            times[name] = elapsed.seconds
            enc.reset()
            mod.reset()
        print(f'{device}: gaines3 {times["gaines3"]:.4f}s vs lin {times["lin"]:.4f}s '
              f'(lin/gaines3 ratio {times["lin"] / times["gaines3"]:.2f}x)')


def _suite_weight(polarity):
    if polarity == 'unipolar':
        row = torch.linspace(0.1, 0.9, 64)
    else:
        row = torch.linspace(-0.75, 0.75, 64)
    return torch.stack((row, row.flip(0)))


def make_operation(polarity, timestep, _device):
    return linear_gaines3(
        _suite_weight(polarity), None,
        {
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'sobol',
        },
    )


def make_values(polarity):
    low, high = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(low, high, 64),)


def make_performance_values(polarity):
    values = make_values(polarity)[0]
    return (values.repeat(2048, 1),)


def analytic_reference(values, polarity):
    return _suite_weight(polarity) @ values[0] / 64


def known_answer_case(polarity):
    values = torch.ones(64)
    return (
        (values,),
        analytic_reference((values,), polarity),
        1.6 / (256 ** 0.5),
    )


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 1.6,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [2],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_linear_gaines3():
    """Verify linear_gaines3 for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_linear_gaines3()

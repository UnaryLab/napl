import time
import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.sim.metric import accuracy
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import (
    devices,
    streaming_suite,
    sync,
)
from napl.sim.module import encoder, linear_pc
from napl.sim.module.linear import linear


class napl_linear_pc(napl_base):
    """Wire encoder -> linear_pc and accumulate the per-timestep PC count."""


    def __init__(self, codec_config, pc_config, weight, bias):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.pc = linear_pc(weight, bias, pc_config)
        self.acc = None


    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        i_spike = self.encoder(input_x)
        count = self.pc(i_spike)
        self.acc = count if self.acc is None else self.acc + count


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.encoder.reset()
        self.pc.reset()
        self.acc = None


def _pc_value(mean_count, polarity, entry):
    """Recover the inner product from the mean PC count (AND-count / XNOR-count form)."""
    if polarity == 'bipolar':
        return 2 * mean_count - entry
    return mean_count


def _kernel_specific_checks():
    """
    The streaming parallel-counter (linear_pc) reproduces (W x + b) within a
    stochastic-computing bound, for both polarities, on every device. Per-timestep the PC
    count lies in [0, entry]; accumulated/T it recovers the inner product (directly for
    unipolar, as 2*mean - entry for bipolar). UnarySim reference: FSULinearPC.
    """
    timestep = 1024
    in_features, out_features = 16, 8
    bound = 3.0 / (timestep ** 0.5)  # Include slack for fan-in SC error.

    for device in devices():
        for polarity in ['unipolar', 'bipolar']:
            for has_bias in [True, False]:
                input_x = gen_rand_tensor(polarity, shape=(in_features,), width=8).type(global_config.ntype).to(device)
                weight = gen_rand_tensor(polarity, shape=(out_features, in_features), width=8).type(global_config.ntype).to(device)
                bias = gen_rand_tensor(polarity, shape=(out_features,), width=8).type(global_config.ntype).to(device) if has_bias else None

                codec_config = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
                pc_config = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 2}

                inst = napl_linear_pc(codec_config, pc_config, weight, bias).to(device)
                inst(input_x, timesteps=timestep)

                entry = in_features + (1 if has_bias else 0)
                ref = weight @ input_x + (bias if has_bias else 0)
                val = _pc_value(inst.acc / timestep, polarity, entry)

                err = (val - ref).abs()
                rmse = torch.sqrt(err.pow(2).mean()).item()
                assert err.max().item() < 0.1, \
                    f'{device}/{polarity}/bias={has_bias}: max err {err.max().item()} too large'
                assert rmse < bound * 3, \
                    f'{device}/{polarity}/bias={has_bias}: rmse {rmse} exceeds bound {bound*3}'
                assert inst.pc.timestep_cur == timestep
                print(f'[{device}] {polarity} bias={has_bias}: rmse={rmse:.5f} max_err={err.max().item():.5f}')
                inst.reset()

    # All-ones unipolar operands produce the exact population count.
    w1_cpu = torch.ones(out_features, in_features).type(global_config.ntype)
    x1_cpu = torch.ones(in_features).type(global_config.ntype)
    for device in devices():
        w1 = w1_cpu.to(device)
        x1 = x1_cpu.to(device)
        inst = napl_linear_pc({'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 1},
                                  {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 2},
                                  w1, None).to(device)
        inst(x1, timesteps=64)
        assert torch.allclose(inst.acc / 64, torch.full((out_features,), float(in_features), device=device)), \
            f'[{device}] all-ones unipolar PC count should equal in_features every step'
        inst.reset()
    print('known-answer corner passed.')

    # Compare kernels on identical inputs; linear_pc omits the scaled accumulator.
    input_x_cpu = gen_rand_tensor('bipolar', shape=(in_features,), width=8).type(global_config.ntype)
    weight_cpu = gen_rand_tensor('bipolar', shape=(out_features, in_features), width=8).type(global_config.ntype)
    bias_cpu = gen_rand_tensor('bipolar', shape=(out_features,), width=8).type(global_config.ntype)
    pc_cfg = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2}
    for device in devices():
        input_x = input_x_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        enc = encoder({'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}).to(device)
        pc = linear_pc(weight, bias, pc_cfg).to(device)
        lin = linear(weight, bias, {**pc_cfg, 'scale': None, 'width': 12}).to(device)

        spikes = [enc(input_x).clone() for _ in range(timestep)]
        enc.reset()
        sync(device)
        t0 = time.time()
        for s in spikes:
            pc(s)
        sync(device)
        t_pc = time.time() - t0
        pc.reset()
        sync(device)
        t0 = time.time()
        for s in spikes:
            lin(s)
        sync(device)
        t_lin = time.time() - t0
        lin.reset()
        print(f'[{device}] perf: linear_pc {t_pc*1e3:.1f}ms vs linear {t_lin*1e3:.1f}ms '
              f'(speedup {t_lin/max(t_pc,1e-9):.2f}x)')

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
    return linear_pc(
        _suite_weight(polarity), None,
        {
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 2,
        },
    )


def make_values(polarity):
    if polarity == 'unipolar':
        return (torch.tensor([0.1, 0.3, 0.5, 0.7]),)
    return (torch.tensor([-0.75, -0.25, 0.25, 0.75]),)


def analytic_reference(values, polarity):
    result = _suite_weight(polarity) @ values[0]
    return (result + 4) / 2 if polarity == 'bipolar' else result


def known_answer_case(polarity):
    values = torch.ones(4)
    return (values,), analytic_reference((values,), polarity), 0.0


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 4.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'make_readout': lambda _polarity, _timestep, _device: accuracy(
        {'polarity': 'unipolar'}
    ),
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_linear_pc():
    """Verify linear_pc against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_linear_pc()

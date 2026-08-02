import time
import torch
import torch.nn.functional as F

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.sim.metric import accuracy
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import (
    devices,
    streaming_suite,
    sync,
)
from napl.sim.module import encoder
from napl.sim.module.conv_pc import conv_pc
from napl.sim.module.conv import conv


class napl_conv_pc(napl_base):
    """Wire encoder -> conv_pc and accumulate the per-timestep PC count."""


    def __init__(self, codec_config, pc_config, weight, bias, stride, padding):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.pc = conv_pc(weight, bias, stride=stride, padding=padding, config=pc_config)
        self.acc = None


    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        count = self.pc(self.encoder(input_x))
        self.acc = count if self.acc is None else self.acc + count


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.encoder.reset()
        self.pc.reset()
        self.acc = None


def _pc_value(mean_count, polarity, entry):
    """Recover the conv result from the mean PC count (AND-count / XNOR-count form)."""
    if polarity == 'bipolar':
        return 2 * mean_count - entry
    return mean_count


def _kernel_specific_checks():
    """
    The streaming conv parallel-counter (conv_pc) reproduces (conv2d(x, W) + b)
    within a stochastic-computing bound, for both polarities, on every device. Per-timestep
    the PC count lies in [0, entry]; accumulated/T it recovers the conv (directly for
    unipolar, as 2*mean - entry for bipolar). padding=1 exercises the decorrelated rate-0.5
    pad stream. UnarySim reference: FSUConv2dPC.
    """
    ntype = global_config.ntype
    timestep = 512
    b, ic, oc, hw, k = 2, 3, 4, 8, 3

    for device in devices():
        for polarity in ['unipolar', 'bipolar']:
            for has_bias in [True, False]:
                for pad in [0, 1]:
                    x = gen_rand_tensor(polarity, (b, ic, hw, hw), 8).type(ntype).to(device)
                    weight = gen_rand_tensor(polarity, (oc, ic, k, k), 8).type(ntype).to(device)
                    bias = gen_rand_tensor(polarity, (oc,), 8).type(ntype).to(device) if has_bias else None

                    codec = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
                    pc_cfg = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 2}

                    inst = napl_conv_pc(codec, pc_cfg, weight, bias, 1, pad).to(device)
                    inst(x, timesteps=timestep)

                    entry = ic * k * k + (1 if has_bias else 0)
                    ref = F.conv2d(x, weight, bias, stride=1, padding=pad)
                    val = _pc_value(inst.acc / timestep, polarity, entry)

                    err = (val - ref).abs()
                    rmse = err.pow(2).mean().sqrt().item()
                    assert val.shape == ref.shape, (device, polarity, has_bias, pad)
                    assert inst.pc.timestep_cur == timestep
                    assert rmse < 0.4, \
                        f'{device}/{polarity}/bias={has_bias}/pad={pad}: rmse {rmse} too large'
                    print(f'[{device}] {polarity} bias={has_bias} pad={pad}: rmse={rmse:.4f} '
                          f'max_err={err.max().item():.4f}')
                    inst.reset()

    # All-ones unipolar operands produce the exact population count without padding.
    for device in devices():
        w1 = torch.ones(oc, ic, k, k).type(ntype).to(device)
        x1 = torch.ones(1, ic, k, k).type(ntype).to(device)
        inst = napl_conv_pc(
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 1},
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 2},
            w1, None, 1, 0,
        ).to(device)
        inst(x1, timesteps=64)
        entry = ic * k * k
        assert torch.allclose(
            inst.acc / 64, torch.full_like(inst.acc, float(entry))
        ), 'all-ones unipolar PC count should equal in*kh*kw every step'
        inst.reset()
        print(f'[{device}] known-answer corner passed.')

    # Compare kernels on identical spikes; conv_pc omits the scaled accumulator.
    perf_x = gen_rand_tensor('bipolar', (b, ic, hw, hw), 8).type(ntype)
    perf_weight = gen_rand_tensor('bipolar', (oc, ic, k, k), 8).type(ntype)
    perf_bias = gen_rand_tensor('bipolar', (oc,), 8).type(ntype)
    for device in devices():
        x = perf_x.to(device)
        weight = perf_weight.to(device)
        bias = perf_bias.to(device)
        pc_cfg = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2}
        enc = encoder({'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}).to(device)
        pc = conv_pc(weight, bias, stride=1, padding=0, config=pc_cfg).to(device)
        full = conv(weight, bias, stride=1, padding=0, config={**pc_cfg, 'width': 12}).to(device)

        spikes = [enc(x).clone() for _ in range(timestep)]
        enc.reset()
        sync(device)
        t0 = time.time()
        for spike in spikes:
            pc(spike)
        sync(device)
        t_pc = time.time() - t0
        pc.reset()
        sync(device)
        t0 = time.time()
        for spike in spikes:
            full(spike)
        sync(device)
        t_full = time.time() - t0
        full.reset()
        print(f'[{device}] perf: conv_pc {t_pc*1e3:.1f}ms vs conv {t_full*1e3:.1f}ms '
              f'(speedup {t_full/max(t_pc,1e-9):.2f}x)')

    print('Test passed.')


def make_operation(polarity, timestep, _device):
    weight = torch.ones(1, 1, 1, 1, dtype=global_config.ntype)
    return conv_pc(
        weight, None, stride=1, padding=0,
        config={
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 2,
        },
    )


def make_values(polarity):
    low = -0.75 if polarity == 'bipolar' else 0.0
    return (torch.linspace(low, 0.75, 16).reshape(1, 1, 4, 4),)


def analytic_reference(values, polarity):
    result = values[0]
    return (result + 1) / 2 if polarity == 'bipolar' else result


def known_answer_case(_polarity):
    values = torch.ones(1, 1, 2, 2)
    return (values,), torch.ones_like(values), 0.0


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 3.0,
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


def test_conv_pc():
    """Verify conv_pc against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_conv_pc()

import time
import torch
import torch.nn.functional as F

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.module import encoder
from napl.module.conv_fsu_pc import conv_fsu_pc
from napl.module.conv import conv_fsu


def _devices():
    devs = ['cpu']
    if torch.cuda.is_available():
        devs.append('cuda')
    if torch.backends.mps.is_available():
        devs.append('mps')
    return devs


class napl_conv_fsu_pc(napl_base):
    """Wire encoder -> conv_fsu_pc and accumulate the per-timestep PC count."""
    def __init__(self, codec_config, pc_config, weight, bias, stride, padding):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.pc = conv_fsu_pc(weight, bias, stride=stride, padding=padding, config=pc_config)
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


def test_conv_fsu_pc():
    """
    The streaming FSU conv parallel-counter (conv_fsu_pc) reproduces (conv2d(x, W) + b)
    within a stochastic-computing bound, for both polarities, on every device. Per-timestep
    the PC count lies in [0, entry]; accumulated/T it recovers the conv (directly for
    unipolar, as 2*mean - entry for bipolar). padding=1 exercises the decorrelated rate-0.5
    pad stream. UnarySim reference: FSUConv2dPC.
    """
    ntype = global_config.ntype
    timestep = 512
    b, ic, oc, hw, k = 2, 3, 4, 8, 3

    for device in _devices():
        for polarity in ['unipolar', 'bipolar']:
            for has_bias in [True, False]:
                for pad in [0, 1]:
                    x = gen_rand_tensor(polarity, (b, ic, hw, hw), 8).type(ntype).to(device)
                    weight = gen_rand_tensor(polarity, (oc, ic, k, k), 8).type(ntype).to(device)
                    bias = gen_rand_tensor(polarity, (oc,), 8).type(ntype).to(device) if has_bias else None

                    codec = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
                    pc_cfg = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 2}

                    inst = napl_conv_fsu_pc(codec, pc_cfg, weight, bias, 1, pad).to(device)
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

    # known-answer corner: unipolar all-ones input & weight, no pad => PC count == entry every step
    device = _devices()[0]
    w1 = torch.ones(oc, ic, k, k).type(ntype).to(device)
    x1 = torch.ones(1, ic, k, k).type(ntype).to(device)
    inst = napl_conv_fsu_pc({'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 1},
                            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 2},
                            w1, None, 1, 0).to(device)
    inst(x1, timesteps=64)
    entry = ic * k * k
    assert torch.allclose(inst.acc / 64, torch.full_like(inst.acc, float(entry))), \
        'all-ones unipolar PC count should equal in*kh*kw every step'
    inst.reset()
    print('known-answer corner passed.')

    # performance: time conv_fsu_pc (no accumulator) vs conv_fsu (with scaled adder) on
    # identical input spikes; the PC kernel should not be slower than the full conv.
    device = _devices()[-1]
    sync = (lambda: torch.cuda.synchronize()) if device == 'cuda' else \
           ((lambda: torch.mps.synchronize()) if device == 'mps' else (lambda: None))
    x = gen_rand_tensor('bipolar', (b, ic, hw, hw), 8).type(ntype).to(device)
    weight = gen_rand_tensor('bipolar', (oc, ic, k, k), 8).type(ntype).to(device)
    bias = gen_rand_tensor('bipolar', (oc,), 8).type(ntype).to(device)
    pc_cfg = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2}
    enc = encoder({'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}).to(device)
    pc = conv_fsu_pc(weight, bias, stride=1, padding=0, config=pc_cfg).to(device)
    full = conv_fsu(weight, bias, stride=1, padding=0, config={**pc_cfg, 'width': 12}).to(device)

    spikes = [enc(x).clone() for _ in range(timestep)]
    enc.reset()
    sync()
    t0 = time.time()
    for s in spikes:
        pc(s)
    sync()
    t_pc = time.time() - t0
    pc.reset()
    sync()
    t0 = time.time()
    for s in spikes:
        full(s)
    sync()
    t_full = time.time() - t0
    full.reset()
    print(f'[{device}] perf: conv_fsu_pc {t_pc*1e3:.1f}ms vs conv_fsu {t_full*1e3:.1f}ms '
          f'(speedup {t_full/max(t_pc,1e-9):.2f}x)')

    print('Test passed.')


if __name__ == '__main__':
    test_conv_fsu_pc()

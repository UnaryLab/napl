import time

import torch
import torch.nn.functional as F

# import directly from the module: not yet wired into napl.sim.module.__init__
from napl.utils._shared_test import devices, sync
from napl.sim.module.mgu_hardpt import mgu_hardpt


def _ref_mgu_pt(x, hx, Wih, bih, Whh, bhh):
    """Direct hard-activation PT-style MGU, independent implementation."""
    gi = F.hardtanh(F.linear(x, Wih, bih))
    gh = F.hardtanh(F.linear(hx, Whh, bhh))
    i_f, i_n = gi.chunk(2, 1)
    h_f, h_n = gh.chunk(2, 1)
    fg = F.hardsigmoid(F.hardtanh(i_f + h_f) * 3)
    ng = F.hardtanh(i_n + fg * h_n)
    return F.hardtanh(ng - fg * ng + fg * hx)


def test_mgu_hardpt():
    """
    Correctness: mgu_hardpt reproduces the PT-style hard-activation MGU equations
    exactly on every device; hx=None default; trainable gradients.
    Performance: cell vs the functional reference on identical inputs, per device.
    """
    torch.manual_seed(0)
    isz, hsz, b = 6, 4, 5
    cell = mgu_hardpt(isz, hsz, bias=True)
    x0 = torch.rand(b, isz) * 2 - 1
    hx0 = torch.rand(b, hsz) * 2 - 1

    for device in devices():
        c = cell.to(device)
        x, hx = x0.to(device), hx0.to(device)
        y = c(x, hx)
        ref = _ref_mgu_pt(x, hx, c.weight_ih, c.bias_ih, c.weight_hh, c.bias_hh)
        assert y.shape == (b, hsz)
        assert torch.allclose(y, ref, atol=1e-6), f'mismatch on {device}'
        assert c(x).shape == (b, hsz)  # hx=None default
        # output stays in the legal unary range
        assert y.abs().max().item() <= 1.0 + 1e-6

        # gradients flow (hard activations are piecewise-linear, autograd-native)
        xg = x.clone().requires_grad_(True)
        c(xg, hx).sum().backward()
        assert torch.isfinite(xg.grad).all()
        assert torch.isfinite(c.weight_ih.grad).all()
        c.zero_grad()

        # performance: cell vs functional reference, identical inputs
        n = 50
        sync(device)
        t0 = time.perf_counter()
        for _ in range(n):
            c(x, hx)
        sync(device)
        t_cell = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n):
            _ref_mgu_pt(x, hx, c.weight_ih, c.bias_ih, c.weight_hh, c.bias_hh)
        sync(device)
        t_ref = time.perf_counter() - t0
        print(f'[{device}] mgu_hardpt {t_cell*1e3/n:.3f} ms/iter, '
              f'reference {t_ref*1e3/n:.3f} ms/iter, ratio {t_ref/max(t_cell,1e-12):.2f}x')

    # soft (hard=False) path uses true sigmoid/tanh
    soft = mgu_hardpt(isz, hsz, bias=True, config={'hard': False})
    assert soft(x0, hx0).shape == (b, hsz)
    print('PASS')


if __name__ == '__main__':
    test_mgu_hardpt()

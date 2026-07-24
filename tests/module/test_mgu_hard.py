import time

import torch
import torch.nn.functional as F

from napl.sim.module import mgu_hard, mgu_hardfxp
from napl.utils._shared_test import devices, sync


def _ref_mgu(x, hx, Wf, bf, Wn, bn):
    """Direct hard-activation MGU, independent of the cell implementation."""
    fg_in = F.hardtanh(F.linear(torch.cat((hx, x), 1), Wf, bf))
    fg = F.hardsigmoid(fg_in * 3)
    fg_hx = fg * hx
    ng = F.hardtanh(F.linear(torch.cat((fg_hx, x), 1), Wn, bn))
    return F.hardtanh(ng - fg * ng + fg_hx)


def test_mgu_hard():
    """
    mgu_hard reproduces the hard-activation MGU equations exactly; mgu_hardfxp approximates
    it within the fixed-point bound; both train (STE) with finite gradients.
    """
    torch.manual_seed(0)
    isz, hsz, b = 6, 4, 5
    cell = mgu_hard(isz, hsz, bias=True)
    x_cpu = torch.rand(b, isz) * 2 - 1
    hx_cpu = torch.rand(b, hsz) * 2 - 1

    for device in devices():
        cell = cell.to(device)
        x = x_cpu.to(device)
        hx = hx_cpu.to(device)
        sync(device)
        start = time.perf_counter()
        y = cell(x, hx)
        sync(device)
        elapsed = time.perf_counter() - start
        sync(device)
        start = time.perf_counter()
        ref = _ref_mgu(
            x, hx, cell.weight_f, cell.bias_f, cell.weight_n, cell.bias_n
        )
        sync(device)
        ref_elapsed = time.perf_counter() - start
        assert y.shape == (b, hsz)
        assert torch.allclose(y, ref, atol=1e-5)
        assert cell(x).shape == (b, hsz)

        cfx = mgu_hardfxp(
            isz, hsz, bias=True, config={'intwidth': 3, 'fracwidth': 6}
        ).to(device)
        for target, source in [
            (cfx.weight_f, cell.weight_f),
            (cfx.weight_n, cell.weight_n),
            (cfx.bias_f, cell.bias_f),
            (cfx.bias_n, cell.bias_n),
        ]:
            target.data = source.data.clone()
        rmse = (cfx(x, hx) - y).pow(2).mean().sqrt().item()
        print(
            f'[{device}] mgu_hardfxp rmse={rmse:.4f}, '
            f'hard/reference ratio={ref_elapsed / max(elapsed, 1e-12):.2f}x'
        )
        assert rmse < 0.05, (device, rmse)

        for module in [cell, cfx]:
            xg = x.clone().requires_grad_(True)
            module(xg, hx).sum().backward()
            assert torch.isfinite(xg.grad).all()
            assert torch.isfinite(module.weight_f.grad).all()
            module.zero_grad()

    print('Test passed.')


if __name__ == '__main__':
    test_mgu_hard()

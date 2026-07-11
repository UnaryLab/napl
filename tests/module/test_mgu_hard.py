import torch
import torch.nn.functional as F

from napl.module import mgu_hard, mgu_hardfxp


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
    x = torch.rand(b, isz) * 2 - 1
    hx = torch.rand(b, hsz) * 2 - 1

    y = cell(x, hx)
    ref = _ref_mgu(x, hx, cell.weight_f, cell.bias_f, cell.weight_n, cell.bias_n)
    assert y.shape == (b, hsz)
    assert torch.allclose(y, ref, atol=1e-5)
    assert cell(x).shape == (b, hsz)   # hx=None default

    # fxp variant approximates the float cell
    cfx = mgu_hardfxp(isz, hsz, bias=True, config={'intwidth': 3, 'fracwidth': 6})
    for a, src in [(cfx.weight_f, cell.weight_f), (cfx.weight_n, cell.weight_n),
                   (cfx.bias_f, cell.bias_f), (cfx.bias_n, cell.bias_n)]:
        a.data = src.data.clone()
    rmse = (cfx(x, hx) - y).pow(2).mean().sqrt().item()
    print(f'mgu_hardfxp vs mgu_hard rmse={rmse:.4f}')
    assert rmse < 0.05, rmse

    # gradients flow (incl. STE through round_fxp)
    for c in [cell, cfx]:
        xg = x.clone().requires_grad_(True)
        c(xg, hx).sum().backward()
        assert torch.isfinite(xg.grad).all()
        assert torch.isfinite(c.weight_f.grad).all()

    print('Test passed.')


if __name__ == '__main__':
    test_mgu_hard()

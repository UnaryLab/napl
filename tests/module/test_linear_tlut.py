import torch
import torch.nn.functional as F

from napl.module import linear_tlut


def test_linear_tlut():
    """
    Binary-domain temporal-LUT linear matches nn.Linear within the temporal-decomposition
    bound across all three modes (fxpfxp, fxpfp, fpfp) and both temporal operands, with
    gradients flowing via the STE.
    """
    torch.manual_seed(0)
    in_features, out_features, batch = 32, 16, 8
    weight = torch.rand(out_features, in_features) * 2 - 1
    bias = torch.rand(out_features) * 2 - 1
    x = torch.rand(batch, in_features) * 2 - 1
    ref = F.linear(x, weight, bias)

    cases = [
        ({'temporal': 'i', 'formati': 'fxp', 'formatw': 'fxp', 'widtht': 4, 'widthi': 8, 'widthw': 8}, 'fxpfxp', 0.05),
        ({'temporal': 'w', 'formati': 'fxp', 'formatw': 'fxp', 'widtht': 4, 'widthi': 8, 'widthw': 8}, 'fxpfxp', 0.05),
        ({'temporal': 'i', 'formati': 'float32', 'formatw': 'fxp', 'widtht': 4}, 'fxpfp', 0.05),
        ({'temporal': 'i', 'formati': 'float32', 'formatw': 'float32', 'widtht': 4}, 'fpfp', 0.05),
    ]
    for cfg, expect_mode, bound in cases:
        lin = linear_tlut(in_features, out_features, bias=True, weight_ext=weight, bias_ext=bias, config=cfg)
        assert lin.mode == expect_mode, (lin.mode, expect_mode)
        y = lin(x)
        rmse = (y - ref).pow(2).mean().sqrt().item()
        print(f'  {expect_mode} temporal={cfg["temporal"]}: rmse={rmse:.5f}')
        assert y.shape == ref.shape
        assert rmse < bound, (expect_mode, rmse)

    xg = x.clone().requires_grad_(True)
    linear_tlut(in_features, out_features, bias=True, weight_ext=weight, bias_ext=bias)(xg).sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()

    print('Test passed.')


if __name__ == '__main__':
    test_linear_tlut()

import torch
import torch.nn.functional as F

from napl.module import linear_hub


def test_linear_hub():
    """
    Binary-domain HUB linear (unary-multiplication value map) matches nn.Linear within
    the unary approximation bound, and the STE lets gradients flow.
    """
    torch.manual_seed(0)
    in_features, out_features, batch = 32, 16, 8
    weight = torch.rand(out_features, in_features) * 2 - 1
    bias = torch.rand(out_features) * 2 - 1
    x = torch.rand(batch, in_features) * 2 - 1

    lin = linear_hub(in_features, out_features, bias=True, weight_ext=weight, bias_ext=bias)
    y = lin(x)
    ref = F.linear(x, weight, bias)
    rmse = (y - ref).pow(2).mean().sqrt().item()
    print(f'linear_hub vs F.linear: rmse={rmse:.5f} max_err={(y-ref).abs().max().item():.5f}')

    assert y.shape == ref.shape
    assert rmse < 0.06, rmse

    xg = x.clone().requires_grad_(True)
    lin(xg).sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()

    print('Test passed.')


if __name__ == '__main__':
    test_linear_hub()

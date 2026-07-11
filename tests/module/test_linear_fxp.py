import torch
import torch.nn.functional as F

from napl.module import linear_fxp


def test_linear_fxp():
    """
    Binary-domain fixed-point linear matches nn.Linear within the 8-bit quantization
    bound, and the STE lets gradients flow to input and weight.
    """
    torch.manual_seed(0)
    in_features, out_features, batch = 32, 16, 8
    weight = torch.rand(out_features, in_features) * 2 - 1
    bias = torch.rand(out_features) * 2 - 1
    x = torch.rand(batch, in_features) * 2 - 1

    lin = linear_fxp(in_features, out_features, bias=True, weight_ext=weight, bias_ext=bias,
                     config={'widthi': 8, 'widthw': 8, 'quantilei': 1, 'quantilew': 1, 'rounding': 'round'})
    y = lin(x)
    ref = F.linear(x, weight, bias)
    rmse = (y - ref).pow(2).mean().sqrt().item()
    print(f'linear_fxp vs F.linear: rmse={rmse:.5f} max_err={(y-ref).abs().max().item():.5f}')

    assert y.shape == ref.shape
    assert rmse < 0.05, rmse

    xg = x.clone().requires_grad_(True)
    lin(xg).sum().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()

    # degenerate all-zero input must stay finite (rshift_offset log2(0) guard), giving the bias
    y_zero = lin(torch.zeros(batch, in_features))
    assert torch.isfinite(y_zero).all()
    assert torch.allclose(y_zero, bias.unsqueeze(0).expand_as(y_zero), atol=1e-5)

    print('Test passed.')


if __name__ == '__main__':
    test_linear_fxp()

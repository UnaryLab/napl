import torch
import torch.nn.functional as F

from napl.module import conv_fxp, conv_hub, conv_tlut


def test_conv_binary():
    """
    Binary-domain conv variants (fxp/hub/tlut) match nn.Conv2d within their quantization
    bounds for padding 0 and 1, and the STE lets gradients flow to input and weight.
    """
    torch.manual_seed(0)
    b, ic, oc, hw, k = 4, 3, 6, 10, 3
    x = torch.rand(b, ic, hw, hw) * 2 - 1
    weight = torch.rand(oc, ic, k, k) * 2 - 1
    bias = torch.rand(oc) * 2 - 1

    builders = {
        'fxp': lambda pad: conv_fxp(ic, oc, k, padding=pad, weight_ext=weight, bias_ext=bias),
        'hub': lambda pad: conv_hub(ic, oc, k, padding=pad, weight_ext=weight, bias_ext=bias),
        'tlut': lambda pad: conv_tlut(ic, oc, k, padding=pad, weight_ext=weight, bias_ext=bias),
    }
    for pad in [0, 1]:
        ref = F.conv2d(x, weight, bias, stride=1, padding=pad)
        for name, build in builders.items():
            y = build(pad)(x)
            rmse = (y - ref).pow(2).mean().sqrt().item()
            print(f'  pad={pad} conv_{name}: rmse={rmse:.4f}')
            assert y.shape == ref.shape
            assert rmse < 0.05, (name, pad, rmse)

    xg = x.clone().requires_grad_(True)
    m = conv_fxp(ic, oc, k, padding=1, weight_ext=weight, bias_ext=bias)
    m(xg).sum().backward()
    assert torch.isfinite(xg.grad).all() and torch.isfinite(m.weight.grad).all()

    print('Test passed.')


if __name__ == '__main__':
    test_conv_binary()

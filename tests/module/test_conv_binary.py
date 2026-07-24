import time

import torch
import torch.nn.functional as F

from napl.sim.module import conv_fxp, conv_hub, conv_tlut
from napl.utils._shared_test import devices, sync


def test_conv_binary():
    """
    Binary-domain conv variants (fxp/hub/tlut) match nn.Conv2d within their quantization
    bounds for padding 0 and 1, and the STE lets gradients flow to input and weight.
    """
    torch.manual_seed(0)
    b, ic, oc, hw, k = 4, 3, 6, 10, 3
    x_cpu = torch.rand(b, ic, hw, hw) * 2 - 1
    weight_cpu = torch.rand(oc, ic, k, k) * 2 - 1
    bias_cpu = torch.rand(oc) * 2 - 1

    for device in devices():
        x = x_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        builders = {
            'fxp': lambda pad: conv_fxp(ic, oc, k, padding=pad, weight_ext=weight, bias_ext=bias).to(device),
            'hub': lambda pad: conv_hub(ic, oc, k, padding=pad, weight_ext=weight, bias_ext=bias).to(device),
            'tlut': lambda pad: conv_tlut(ic, oc, k, padding=pad, weight_ext=weight, bias_ext=bias).to(device),
        }
        for pad in [0, 1]:
            sync(device)
            start = time.perf_counter()
            ref = F.conv2d(x, weight, bias, stride=1, padding=pad)
            sync(device)
            ref_elapsed = time.perf_counter() - start
            for name, build in builders.items():
                module = build(pad)
                sync(device)
                start = time.perf_counter()
                y = module(x)
                sync(device)
                elapsed = time.perf_counter() - start
                rmse = (y - ref).pow(2).mean().sqrt().item()
                print(
                    f'[{device}] pad={pad} conv_{name}: rmse={rmse:.4f}, '
                    f'ratio={ref_elapsed / max(elapsed, 1e-12):.2f}x'
                )
                assert y.shape == ref.shape
                assert rmse < 0.05, (device, name, pad, rmse)

        xg = x.clone().requires_grad_(True)
        m = conv_fxp(
            ic, oc, k, padding=1, weight_ext=weight, bias_ext=bias
        ).to(device)
        m(xg).sum().backward()
        assert torch.isfinite(xg.grad).all() and torch.isfinite(m.weight.grad).all()

    print('Test passed.')


if __name__ == '__main__':
    test_conv_binary()

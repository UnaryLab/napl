import time

import torch
import torch.nn.functional as F

from napl.sim.module import linear_hub
from napl.utils._shared_test import devices, sync


def test_linear_hub():
    """
    Binary-domain HUB linear (unary-multiplication value map) matches nn.Linear within
    the unary approximation bound, and the STE lets gradients flow.
    """
    torch.manual_seed(0)
    in_features, out_features, batch = 32, 16, 8
    weight_cpu = torch.rand(out_features, in_features) * 2 - 1
    bias_cpu = torch.rand(out_features) * 2 - 1
    x_cpu = torch.rand(batch, in_features) * 2 - 1

    for device in devices():
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        x = x_cpu.to(device)
        lin = linear_hub(
            in_features,
            out_features,
            bias=True,
            weight_ext=weight,
            bias_ext=bias,
        ).to(device)
        sync(device)
        start = time.perf_counter()
        y = lin(x)
        sync(device)
        elapsed = time.perf_counter() - start
        sync(device)
        start = time.perf_counter()
        ref = F.linear(x, weight, bias)
        sync(device)
        ref_elapsed = time.perf_counter() - start
        rmse = (y - ref).pow(2).mean().sqrt().item()
        print(
            f'[{device}] linear_hub rmse={rmse:.5f}, '
            f'max={(y-ref).abs().max().item():.5f}, '
            f'ratio={ref_elapsed / max(elapsed, 1e-12):.2f}x'
        )
        assert y.shape == ref.shape
        assert rmse < 0.06, (device, rmse)

        xg = x.clone().requires_grad_(True)
        lin(xg).sum().backward()
        assert xg.grad is not None and torch.isfinite(xg.grad).all()
        assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()

    print('Test passed.')


if __name__ == '__main__':
    test_linear_hub()

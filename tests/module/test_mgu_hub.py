import time

import torch

from napl.sim.base import global_config
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import mgu_hard, mgu_hub


def test_mgu_hub():
    """Verify mgu_hub decodes close to mgu_hard within the stochastic error bound."""
    ntype = global_config.ntype
    torch.manual_seed(0)
    isz, hsz, b = 6, 4, 3

    ref = mgu_hard(isz, hsz, bias=True)
    x_cpu = gen_rand_tensor('bipolar', (b, isz), 8).type(ntype)
    hx_cpu = gen_rand_tensor('bipolar', (b, hsz), 8).type(ntype)

    for device in devices():
        ref = ref.to(device)
        x = x_cpu.to(device)
        hx = hx_cpu.to(device)
        sync(device)
        start = time.perf_counter()
        y_ref = ref(x, hx)
        sync(device)
        ref_elapsed = time.perf_counter() - start
        hub = mgu_hub(
            isz,
            hsz,
            bias=True,
            weight_f=ref.weight_f.data,
            bias_f=ref.bias_f.data,
            weight_n=ref.weight_n.data,
            bias_n=ref.bias_n.data,
            config={'polarity': 'bipolar', 'width': 8, 'generator': 'sobol'},
        ).to(device)
        sync(device)
        start = time.perf_counter()
        y_hub = hub(x, hx)
        sync(device)
        elapsed = time.perf_counter() - start
        rmse = (y_hub - y_ref).pow(2).mean().sqrt().item()
        print(
            f'[{device}] mgu_hub rmse={rmse:.4f}, '
            f'hard/hub ratio={ref_elapsed / max(elapsed, 1e-12):.2f}x'
        )

        assert y_hub.shape == y_ref.shape
        assert rmse < 0.15, (device, rmse)
        assert hub(x).shape == (b, hsz)

    print('Test passed.')


if __name__ == '__main__':
    test_mgu_hub()

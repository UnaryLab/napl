import torch

from napl.utils import (num2tuple, rshift_offset, nn_weight_unary_clip,
                        pow2_lshift, pow2_rshift)
from napl.utils._shared_test import devices, timer


def test_utils():
    """Verify shared synchronization, power-of-two, and tensor utility contracts."""
    assert num2tuple(3) == (3, 3)
    assert num2tuple((2, 4)) == (2, 4)

    for device in devices():
        with timer(device) as elapsed:
            x = torch.tensor([1.5, -2.0], device=device)
            assert torch.allclose(pow2_lshift(x, 2), x * 4)
            assert torch.allclose(pow2_rshift(x, 1), x / 2)

            inp = torch.randn(64, device=device) * 10
            wgt = torch.randn(32, 64, device=device) * 0.1
            ri, _rw, _ro = rshift_offset(inp, wgt, 7, 7, 'round', 1, 1)
            recon = pow2_lshift(pow2_rshift(inp, ri), ri)
            assert torch.allclose(recon, inp, atol=1e-4)

            zi, zw, zo = rshift_offset(
                torch.zeros(64, device=device), wgt, 7, 7, 'round', 1, 1
            )
            assert torch.isfinite(
                torch.tensor([float(zi), float(zw), float(zo)], device=device)
            ).all(), (device, zi, zw, zo)

            lin = torch.nn.Linear(8, 4).to(device)
            with torch.no_grad():
                lin.weight.mul_(100)
            nn_weight_unary_clip(
                frequency=2, mode='bipolar', method='clip', bitwidth=8
            )(lin)
            assert lin.weight.abs().max() <= 1.0
        print(f'[{device}] time={elapsed.seconds * 1000:.1f}ms')

    print('Test passed.')


if __name__ == '__main__':
    test_utils()

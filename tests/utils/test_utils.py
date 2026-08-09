import torch

from napl.utils import (num2tuple, nn_weight_unary_clip,
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

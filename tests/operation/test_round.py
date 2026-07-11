import torch

from napl.operation import round_fxp, round_ste


def test_round_fxp():
    """
    round_fxp quantizes to a fixed-point grid; round_ste passes gradients through.
    """
    q = round_fxp({'intwidth': 3, 'fracwidth': 4})   # grid = 1/16, range [1-128, 127]/16

    x = torch.tensor([0.1, 0.5, -0.3, 1.0])
    y = q(x)
    expected = torch.round(x * 16) / 16
    assert torch.allclose(y, expected), f'{y} != {expected}'

    # clamping: a value above the max grid point saturates at max_val / 2**fracwidth
    big = q(torch.tensor([100.0]))
    assert big.item() == (2 ** (3 + 4) - 1) / 2 ** 4, big

    # straight-through estimator: gradient is identity through the round
    xg = torch.tensor([0.13, -0.42], requires_grad=True)
    round_ste(xg, fracwidth=4).sum().backward()
    assert torch.allclose(xg.grad, torch.ones_like(xg)), xg.grad

    print('Test passed.')


if __name__ == '__main__':
    test_round_fxp()

import torch

from napl.utils import (num2tuple, conv2d_output_shape, conv2d_get_padding,
                        rshift_offset, NN_SC_Weight_Clipper, pow2_lshift, pow2_rshift)


def test_utils():
    # num2tuple
    assert num2tuple(3) == (3, 3)
    assert num2tuple((2, 4)) == (2, 4)

    # conv2d_output_shape matches a real nn.Conv2d
    for ksize, stride, pad in [(3, 1, 1), (5, 2, 0), (3, 2, 1)]:
        conv = torch.nn.Conv2d(2, 4, ksize, stride=stride, padding=pad)
        out = conv(torch.zeros(1, 2, 32, 32))
        assert conv2d_output_shape((32, 32), ksize, stride, pad) == (out.shape[2], out.shape[3]), (ksize, stride, pad)

    # conv2d_get_padding round-trips: padding to keep 32x32 with a 3x3 stride-1 kernel is (1,1),(1,1)
    assert conv2d_get_padding((32, 32), (32, 32), 3, 1) == ((1, 1), (1, 1))

    # pow2 shifts on floats
    x = torch.tensor([1.5, -2.0])
    assert torch.allclose(pow2_lshift(x, 2), x * 4)
    assert torch.allclose(pow2_rshift(x, 1), x / 2)

    # rshift_offset undoes its own scaling: (input >> ri) << ri ~= input (up to rounding)
    inp = torch.randn(64) * 10
    wgt = torch.randn(32, 64) * 0.1
    ri, rw, ro = rshift_offset(inp, wgt, 7, 7, 'round', 1, 1)
    recon = pow2_lshift(pow2_rshift(inp, ri), ri)
    assert torch.allclose(recon, inp, atol=1e-4)

    # degenerate all-zero operand: scale 0 -> log2(0) = -inf must be guarded to finite shifts
    zi, zw, zo = rshift_offset(torch.zeros(64), wgt, 7, 7, 'round', 1, 1)
    assert torch.isfinite(torch.tensor([float(zi), float(zw), float(zo)])).all(), (zi, zw, zo)

    # NN_SC_Weight_Clipper pulls weights into [-1, 1] and quantizes
    lin = torch.nn.Linear(8, 4)
    with torch.no_grad():
        lin.weight.mul_(100)   # push out of range
    NN_SC_Weight_Clipper(frequency=2, mode='bipolar', method='clip', bitwidth=8)(lin)
    assert lin.weight.abs().max() <= 1.0

    print('Test passed.')


if __name__ == '__main__':
    test_utils()

import torch
import torch.nn.functional as F

from napl.operation import relu_hub, sigmoid_hub, tanh_hub


def test_activation_hub():
    """
    Binary-domain activations match their torch piecewise-linear references, pass through
    the [-1,1] / [0,scale] band, and clip outside it.
    """
    x = torch.linspace(-3, 3, 25)

    assert torch.allclose(relu_hub()(x), F.hardtanh(x, 0.0, 1.0))
    assert torch.allclose(relu_hub({'scale': 6.0})(x), F.hardtanh(x, 0.0, 6.0))
    assert torch.allclose(sigmoid_hub()(x), F.hardsigmoid(x * 3))
    assert torch.allclose(tanh_hub()(x), F.hardtanh(x, -1.0, 1.0))

    # known-answer: clipping behavior
    assert torch.allclose(relu_hub()(torch.tensor([-0.5, 0.5, 2.0])), torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(tanh_hub()(torch.tensor([-2.0, 0.3, 2.0])), torch.tensor([-1.0, 0.3, 1.0]))

    # gradients flow through the linear band
    xg = torch.tensor([0.5], requires_grad=True)
    relu_hub()(xg).backward()
    assert xg.grad.item() == 1.0

    print('Test passed.')


if __name__ == '__main__':
    test_activation_hub()

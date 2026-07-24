import time

import torch
import torch.nn.functional as F

from napl.sim.operation import relu_hub, sigmoid_hub, tanh_hub
from napl.utils._shared_test import devices, sync


def test_activation_hub():
    """
    Binary-domain activations match their torch piecewise-linear references, pass through
    the [-1,1] / [0,scale] band, and clip outside it.
    """
    x_cpu = torch.linspace(-3, 3, 25)
    known_relu_cpu = torch.tensor([-0.5, 0.5, 2.0])
    known_tanh_cpu = torch.tensor([-2.0, 0.3, 2.0])

    for device in devices():
        x = x_cpu.to(device)
        relu = relu_hub().to(device)
        relu_scaled = relu_hub({'scale': 6.0}).to(device)
        sigmoid = sigmoid_hub().to(device)
        tanh = tanh_hub().to(device)

        sync(device)
        start = time.perf_counter()
        relu_result = relu(x)
        relu_scaled_result = relu_scaled(x)
        sigmoid_result = sigmoid(x)
        tanh_result = tanh(x)
        sync(device)
        elapsed = time.perf_counter() - start

        sync(device)
        start = time.perf_counter()
        F.hardtanh(x, 0.0, 1.0)
        F.hardtanh(x, 0.0, 6.0)
        F.hardsigmoid(x * 3)
        F.hardtanh(x, -1.0, 1.0)
        sync(device)
        reference_elapsed = time.perf_counter() - start

        assert torch.allclose(relu_result, F.hardtanh(x, 0.0, 1.0))
        assert torch.allclose(relu_scaled_result, F.hardtanh(x, 0.0, 6.0))
        assert torch.allclose(sigmoid_result, F.hardsigmoid(x * 3))
        assert torch.allclose(tanh_result, F.hardtanh(x, -1.0, 1.0))

        known_relu = known_relu_cpu.to(device)
        known_tanh = known_tanh_cpu.to(device)
        assert torch.allclose(relu(known_relu), torch.tensor([0.0, 0.5, 1.0], device=device))
        assert torch.allclose(tanh(known_tanh), torch.tensor([-1.0, 0.3, 1.0], device=device))

        xg = torch.tensor([0.5], device=device, requires_grad=True)
        relu(xg).backward()
        assert xg.grad.item() == 1.0
        ratio = reference_elapsed / elapsed
        print(f'[{device}] kernel={elapsed * 1000:.3f}ms, '
              f'reference={reference_elapsed * 1000:.3f}ms, ratio={ratio:.2f}x')

    print('Test passed.')


if __name__ == '__main__':
    test_activation_hub()

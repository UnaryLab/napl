import time

import torch
import torch.nn.functional as F

from napl.sim.operation import relu_hub
from napl.utils._shared_test import devices, single_shot_suite, sync


def _kernel_specific_checks():
    """
    relu_hub matches torch hardtanh, passes through the [0, scale] band, and clips outside it.
    """
    x_cpu = torch.linspace(-3, 3, 25)
    known_relu_cpu = torch.tensor([-0.5, 0.5, 2.0])

    for device in devices():
        x = x_cpu.to(device)
        relu = relu_hub().to(device)
        relu_scaled = relu_hub({'scale': 6.0}).to(device)

        sync(device)
        start = time.perf_counter()
        relu_result = relu(x)
        relu_scaled_result = relu_scaled(x)
        sync(device)
        elapsed = time.perf_counter() - start

        sync(device)
        start = time.perf_counter()
        F.hardtanh(x, 0.0, 1.0)
        F.hardtanh(x, 0.0, 6.0)
        sync(device)
        reference_elapsed = time.perf_counter() - start

        assert torch.allclose(relu_result, F.hardtanh(x, 0.0, 1.0))
        assert torch.allclose(relu_scaled_result, F.hardtanh(x, 0.0, 6.0))

        known_relu = known_relu_cpu.to(device)
        assert torch.allclose(
            relu(known_relu),
            torch.tensor([0.0, 0.5, 1.0], device=device),
        )

        xg = torch.tensor([0.5], device=device, requires_grad=True)
        relu(xg).backward()
        assert xg.grad.item() == 1.0
        ratio = reference_elapsed / elapsed
        print(
            f'[{device}] kernel={elapsed * 1000:.3f}ms, '
            f'reference={reference_elapsed * 1000:.3f}ms, ratio={ratio:.2f}x'
        )

    print('Test passed.')


def make_module_pair():
    return relu_hub(), torch.nn.Hardtanh(0.0, 1.0)


def make_inputs():
    return (torch.linspace(-3.0, 3.0, 257),)


def known_answer_case():
    values = torch.tensor([-0.5, 0.5, 2.0])
    return relu_hub(), (values,), torch.tensor([0.0, 0.5, 1.0])


def gradient_case():
    return relu_hub(), (torch.tensor([0.5]),)


def expected_ste_gradients(_candidate, _inputs, grad_output):
    return (grad_output,), {}


CONFIG = {
    'quantization_atol': 0.0,
    'known_answer_atol': 0.0,
    'gradient_atol': 0.0,
    'gradient_rtol': 0.0,
    'make_module_pair': make_module_pair,
    'make_inputs': make_inputs,
    'known_answer_case': known_answer_case,
    'gradient_case': gradient_case,
    'expected_ste_gradients': expected_ste_gradients,
    'extra_checks': _kernel_specific_checks,
}


def test_relu_hub():
    """Verify relu_hub quantization and STE gradients against its reference, including timing."""
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_relu_hub()

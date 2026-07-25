import time

import torch
import torch.nn.functional as F

from napl.sim.operation import tanh_hub
from napl.utils._shared_test import devices, single_shot_suite, sync


def _kernel_specific_checks():
    """
    tanh_hub matches torch hardtanh, passes through the [-1, 1] band, and clips outside it.
    """
    x_cpu = torch.linspace(-3, 3, 25)
    known_tanh_cpu = torch.tensor([-2.0, 0.3, 2.0])

    for device in devices():
        x = x_cpu.to(device)
        tanh = tanh_hub().to(device)

        sync(device)
        start = time.perf_counter()
        tanh_result = tanh(x)
        sync(device)
        elapsed = time.perf_counter() - start

        sync(device)
        start = time.perf_counter()
        F.hardtanh(x, -1.0, 1.0)
        sync(device)
        reference_elapsed = time.perf_counter() - start

        assert torch.allclose(tanh_result, F.hardtanh(x, -1.0, 1.0))

        known_tanh = known_tanh_cpu.to(device)
        assert torch.allclose(
            tanh(known_tanh),
            torch.tensor([-1.0, 0.3, 1.0], device=device),
        )

        ratio = reference_elapsed / elapsed
        print(
            f'[{device}] kernel={elapsed * 1000:.3f}ms, '
            f'reference={reference_elapsed * 1000:.3f}ms, ratio={ratio:.2f}x'
        )

    print('Test passed.')


def make_module_pair():
    return tanh_hub(), torch.nn.Hardtanh(-1.0, 1.0)


def make_inputs():
    return (torch.linspace(-3.0, 3.0, 257),)


def known_answer_case():
    values = torch.tensor([-2.0, 0.3, 2.0])
    return tanh_hub(), (values,), torch.tensor([-1.0, 0.3, 1.0])


def gradient_case():
    return tanh_hub(), (torch.tensor([0.3]),)


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


def test_tanh_hub():
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_tanh_hub()

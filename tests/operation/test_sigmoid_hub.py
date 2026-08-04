import torch
import torch.nn.functional as F

from napl.sim.operation import sigmoid_hub
from napl.utils._shared_test import devices, single_shot_suite, timer


def _kernel_specific_checks():
    """
    sigmoid_hub matches the torch piecewise-linear reference.
    """
    x_cpu = torch.linspace(-3, 3, 25)

    for device in devices():
        x = x_cpu.to(device)
        sigmoid = sigmoid_hub().to(device)

        with timer(device) as elapsed:
            sigmoid_result = sigmoid(x)

        with timer(device) as reference_elapsed:
            F.hardsigmoid(x * 3)

        assert torch.allclose(sigmoid_result, F.hardsigmoid(x * 3))

        ratio = reference_elapsed.seconds / elapsed.seconds
        print(
            f'[{device}] kernel={elapsed.seconds * 1000:.3f}ms, '
            f'reference={reference_elapsed.seconds * 1000:.3f}ms, ratio={ratio:.2f}x'
        )

    print('Test passed.')


class sigmoid_reference(torch.nn.Module):
    def forward(self, input):
        return F.hardsigmoid(input * 3)


def make_module_pair():
    return sigmoid_hub(), sigmoid_reference()


def make_inputs():
    return (torch.linspace(-3.0, 3.0, 257),)


def make_performance_values():
    return (make_inputs()[0].repeat(512),)


def known_answer_case():
    values = torch.tensor([-1.0, 0.0, 1.0])
    return sigmoid_hub(), (values,), F.hardsigmoid(values * 3)


def gradient_case():
    return sigmoid_hub(), (torch.tensor([0.0]),)


def expected_ste_gradients(_candidate, _inputs, grad_output):
    return (grad_output * 0.5,), {}


CONFIG = {
    'quantization_atol': 0.0,
    'known_answer_atol': 0.0,
    'gradient_atol': 0.0,
    'gradient_rtol': 0.0,
    'make_module_pair': make_module_pair,
    'make_inputs': make_inputs,
    'make_performance_values': make_performance_values,
    'known_answer_case': known_answer_case,
    'gradient_case': gradient_case,
    'expected_ste_gradients': expected_ste_gradients,
    'extra_checks': _kernel_specific_checks,
}


def test_sigmoid_hub():
    """Verify sigmoid_hub quantization and STE gradients against its reference, including timing."""
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_sigmoid_hub()

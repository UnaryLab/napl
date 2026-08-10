import torch

from napl.sim.module import round_fxp
from napl.utils._shared_test import single_shot_suite


INTWIDTH = 3
FRACWIDTH = 4
MIN_CODE = -(2 ** (INTWIDTH + FRACWIDTH))
MAX_CODE = 2 ** (INTWIDTH + FRACWIDTH) - 1


class round_reference(torch.nn.Module):
    def forward(self, input):
        return (
            torch.round(input * 2**FRACWIDTH)
            .clamp(MIN_CODE, MAX_CODE)
            / 2**FRACWIDTH
        )


def make_module_pair():
    return (
        round_fxp({'intwidth': INTWIDTH, 'fracwidth': FRACWIDTH}),
        round_reference(),
    )


def make_inputs():
    return (torch.linspace(-10.0, 10.0, 10001),)


def make_performance_values():
    return (make_inputs()[0].repeat(11),)


def known_answer_case():
    candidate, reference = make_module_pair()
    input = torch.tensor([0.1, 0.5, -0.3, 1.0, -100.0, 100.0])
    return candidate, (input,), reference(input)


def gradient_case():
    return (
        round_fxp({'intwidth': INTWIDTH, 'fracwidth': FRACWIDTH}),
        (torch.tensor([0.13, -0.42]),),
    )


def expected_ste_gradients(candidate, inputs, grad_output):
    return (grad_output,), {}


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
}


def test_round_fxp():
    """Verify round_fxp quantization and STE gradients against its reference, including timing."""
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_round_fxp()

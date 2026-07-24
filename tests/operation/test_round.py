import torch

from napl.sim.operation import round_fxp, round_ste
from napl.utils._shared_test import devices, single_shot_suite


INTWIDTH = 3
FRACWIDTH = 4
MIN_CODE = 1 - 2 ** (INTWIDTH + FRACWIDTH)
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
    'known_answer_case': known_answer_case,
    'gradient_case': gradient_case,
    'expected_ste_gradients': expected_ste_gradients,
}


def test_round_fxp():
    single_shot_suite(CONFIG)


def test_round_ste_dtype_and_boundaries():
    for device in devices():
        input = torch.tensor(
            [-100.0, -0.3, 0.1, 100.0],
            dtype=torch.float32,
            device=device,
        )
        result = round_ste(
            input,
            fracwidth=FRACWIDTH,
            min_val=MIN_CODE,
            max_val=MAX_CODE,
        )
        expected = round_reference().to(device)(input)
        assert result.dtype == input.dtype
        assert torch.equal(result, expected)


if __name__ == '__main__':
    test_round_fxp()
    test_round_ste_dtype_and_boundaries()

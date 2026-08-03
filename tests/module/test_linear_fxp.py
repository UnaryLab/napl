import torch

from napl.sim.module import linear_fxp
from napl.utils._shared_test import devices, single_shot_suite


_IN_FEATURES = 16
_OUT_FEATURES = 8
_BATCH = 6


def _parameters():
    weight = torch.linspace(
        -0.75, 0.75, _OUT_FEATURES * _IN_FEATURES
    ).reshape(_OUT_FEATURES, _IN_FEATURES)
    bias = torch.linspace(-0.25, 0.25, _OUT_FEATURES)
    return weight, bias


def _candidate():
    weight, bias = _parameters()
    return linear_fxp(
        _IN_FEATURES,
        _OUT_FEATURES,
        weight_ext=weight,
        bias_ext=bias,
        config={
            'widthi': 8,
            'widthw': 8,
            'quantilei': 1,
            'quantilew': 1,
            'rounding': 'round',
        },
    )


def make_module_pair():
    weight, bias = _parameters()
    reference = torch.nn.Linear(_IN_FEATURES, _OUT_FEATURES)
    with torch.no_grad():
        reference.weight.copy_(weight)
        reference.bias.copy_(bias)
    return _candidate(), reference


def make_inputs():
    value = torch.linspace(
        -0.9, 0.9, _BATCH * _IN_FEATURES
    ).reshape(_BATCH, _IN_FEATURES)
    return (value,)


def known_answer_case():
    _, bias = _parameters()
    value = torch.zeros(_BATCH, _IN_FEATURES)
    return _candidate(), (value,), bias.expand(_BATCH, -1)


def gradient_case():
    return _candidate(), make_inputs()


def expected_ste_gradients(candidate, inputs, grad_output):
    value, = inputs
    return (
        (grad_output.matmul(candidate.weight),),
        {
            'weight': grad_output.t().matmul(value),
            'bias': grad_output.sum(0),
        },
    )


CONFIG = {
    'quantization_atol': 0.05,
    'known_answer_atol': 0.0,
    'gradient_atol': 1e-6,
    'gradient_rtol': 1e-6,
    'make_module_pair': make_module_pair,
    'make_inputs': make_inputs,
    'known_answer_case': known_answer_case,
    'gradient_case': gradient_case,
    'expected_ste_gradients': expected_ste_gradients,
}


def test_linear_fxp():
    """Verify linear_fxp quantization and STE gradients against its reference, including timing."""
    single_shot_suite(CONFIG)


def test_fxp_signed_saturation():
    """Verify signed int8 saturation against the FxpLinear code-point contract."""
    weights = torch.tensor([[1.0] * 8, [-1.0] * 8])
    bias = torch.zeros(2)
    inputs = torch.tensor([[2.0] * 8, [-2.0] * 8])
    input_codes = torch.tensor([[127.0] * 8, [-128.0] * 8])
    weight_codes = torch.tensor([[127.0] * 8, [-128.0] * 8])
    expected = input_codes.matmul(weight_codes.t()) / (2 ** 13)

    for device in devices():
        candidate = linear_fxp(
            8,
            2,
            weight_ext=weights,
            bias_ext=bias,
            config={
                'widthi': 8,
                'widthw': 8,
                'quantilei': 1,
                'quantilew': 1,
                'rounding': 'round',
            },
        ).to(device)
        result = candidate(inputs.to(device))
        torch.testing.assert_close(result, expected.to(device), atol=0.0, rtol=0.0)


if __name__ == '__main__':
    test_linear_fxp()
    test_fxp_signed_saturation()

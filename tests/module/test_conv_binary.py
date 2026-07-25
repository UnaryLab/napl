import time

import torch
import torch.nn.functional as F

from napl.sim.module import conv_fxp, conv_hub, conv_tlut
from napl.utils._shared_test import devices, single_shot_suite, sync


def _kernel_specific_checks():
    """
    Binary-domain conv variants (fxp/hub/tlut) match nn.Conv2d within their quantization
    bounds for padding 0 and 1, and the STE lets gradients flow to input and weight.
    """
    torch.manual_seed(0)
    b, ic, oc, hw, k = 4, 3, 6, 10, 3
    x_cpu = torch.rand(b, ic, hw, hw) * 2 - 1
    weight_cpu = torch.rand(oc, ic, k, k) * 2 - 1
    bias_cpu = torch.rand(oc) * 2 - 1

    for device in devices():
        x = x_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        builders = {
            'fxp': lambda pad: conv_fxp(ic, oc, k, padding=pad, weight_ext=weight, bias_ext=bias).to(device),
            'hub': lambda pad: conv_hub(ic, oc, k, padding=pad, weight_ext=weight, bias_ext=bias).to(device),
            'tlut': lambda pad: conv_tlut(ic, oc, k, padding=pad, weight_ext=weight, bias_ext=bias).to(device),
        }
        for pad in [0, 1]:
            sync(device)
            start = time.perf_counter()
            ref = F.conv2d(x, weight, bias, stride=1, padding=pad)
            sync(device)
            ref_elapsed = time.perf_counter() - start
            for name, build in builders.items():
                module = build(pad)
                sync(device)
                start = time.perf_counter()
                y = module(x)
                sync(device)
                elapsed = time.perf_counter() - start
                rmse = (y - ref).pow(2).mean().sqrt().item()
                print(
                    f'[{device}] pad={pad} conv_{name}: rmse={rmse:.4f}, '
                    f'ratio={ref_elapsed / max(elapsed, 1e-12):.2f}x'
                )
                assert y.shape == ref.shape
                assert rmse < 0.05, (device, name, pad, rmse)

        xg = x.clone().requires_grad_(True)
        m = conv_fxp(
            ic, oc, k, padding=1, weight_ext=weight, bias_ext=bias
        ).to(device)
        m(xg).sum().backward()
        assert torch.isfinite(xg.grad).all() and torch.isfinite(m.weight.grad).all()

    print('Test passed.')


class conv_reference(torch.nn.Module):
    def __init__(self, weight, bias, padding):
        super().__init__()
        self.register_buffer('weight', weight.detach().clone())
        self.register_buffer('bias', bias.detach().clone())
        self.padding = padding

    def forward(self, input):
        return F.conv2d(
            input, self.weight, self.bias, stride=1, padding=self.padding
        )


def _make_candidate(padding=1):
    weight = torch.tensor([[[[0.5, -0.25], [0.25, 0.5]]]])
    bias = torch.tensor([0.25])
    candidate = conv_fxp(
        1, 1, 2, padding=padding, weight_ext=weight, bias_ext=bias
    )
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    return candidate


def make_module_pair():
    candidate = _make_candidate()
    return candidate, conv_reference(
        candidate.weight, candidate.bias, candidate.padding
    )


def make_inputs():
    return (torch.linspace(-0.75, 0.75, 25).reshape(1, 1, 5, 5),)


def known_answer_case():
    candidate, reference = make_module_pair()
    values = torch.ones(1, 1, 3, 3) * 0.5
    return candidate, (values,), reference(values)


def gradient_case():
    return _make_candidate(), (torch.linspace(-0.5, 0.5, 16).reshape(1, 1, 4, 4),)


def expected_ste_gradients(candidate, inputs, grad_output):
    input_ref = inputs[0].detach().clone().requires_grad_(True)
    output = F.conv2d(
        input_ref,
        candidate.weight.detach(),
        candidate.bias.detach(),
        stride=1,
        padding=candidate.padding,
    )
    gradient, = torch.autograd.grad(output, input_ref, grad_output)
    return (gradient,), {}


CONFIG = {
    'quantization_atol': 0.05,
    'known_answer_atol': 0.05,
    'gradient_atol': 1e-6,
    'gradient_rtol': 1e-6,
    'make_module_pair': make_module_pair,
    'make_inputs': make_inputs,
    'known_answer_case': known_answer_case,
    'gradient_case': gradient_case,
    'expected_ste_gradients': expected_ste_gradients,
    'extra_checks': _kernel_specific_checks,
}


def test_conv_binary():
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_conv_binary()

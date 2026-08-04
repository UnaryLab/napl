import torch
import torch.nn.functional as F

from napl.sim.module import conv_fxp, conv_hub, conv_tlut
from napl.utils import conv2d_output_shape, pow2_rshift, rshift_offset
from napl.utils._shared_test import devices, single_shot_suite, timer


def _binary_conv_reference(module, input, linear_reference):
    output_size = conv2d_output_shape(
        input.shape[-2:], module.kernel_size, module.stride, module.padding, module.dilation
    )
    columns = F.unfold(input, module.kernel_size, module.dilation, module.padding, module.stride)
    patches = columns.transpose(1, 2).reshape(-1, columns.size(1))
    output = linear_reference(patches, module.weight.view(module.weight.size(0), -1))
    output = output.reshape(input.size(0), -1, output.size(-1)).transpose(1, 2)
    output = output.reshape(input.size(0), output.size(1), *output_size)
    return output + module.bias.view(1, -1, 1, 1)


def test_conv_unarysim_quantization_semantics():
    """Verify conv_fxp follows UnarySim operand, bias, and accumulator quantization semantics."""
    input_fxp = torch.linspace(-0.34, 0.33, 64).reshape(2, 2, 4, 4)
    weight_fxp = torch.linspace(-0.34, 0.32, 36).reshape(2, 2, 3, 3)
    bias_fxp = torch.tensor([0.05, -0.075])
    fxp = conv_fxp(
        2, 2, 3, padding=1, weight_ext=weight_fxp, bias_ext=bias_fxp,
        config={'widthi': 4, 'quantilei': 1, 'widthw': 4,
                'quantilew': 1, 'rounding': 'round'},
    )
    rshift_i, rshift_w, _ = rshift_offset(input_fxp, weight_fxp, 3, 3, 'round')

    def fxp_reference(input, weight):
        input = pow2_rshift(input, rshift_i).round().clamp(-8, 7)
        weight = pow2_rshift(weight, rshift_w).round().clamp(-8, 7)
        return pow2_rshift(input @ weight.t(), -rshift_i - rshift_w)

    assert torch.equal(fxp(input_fxp), _binary_conv_reference(fxp, input_fxp, fxp_reference))

    input_hub = torch.linspace(-0.49, 0.47, 64).reshape(2, 2, 4, 4)
    weight_hub = torch.linspace(-0.46, 0.49, 36).reshape(2, 2, 3, 3)
    bias_hub = torch.tensor([0.025, -0.04])
    hub = conv_hub(
        2, 2, 3, padding=1, weight_ext=weight_hub, bias_ext=bias_hub,
        config={'widthi': 4, 'rngi': 'sobol', 'quantilei': 1,
                'widthw': 4, 'rngw': 'sobol', 'quantilew': 1,
                'cycle': 8, 'rounding': 'round'},
    )
    rshift_i, rshift_w, rshift_o = rshift_offset(input_hub, weight_hub, 3, 3, 'round')

    def hub_reference(input, weight):
        input_index = pow2_rshift(input, rshift_i).abs().long().clamp(0, 7).unsqueeze(1)
        weight_index = pow2_rshift(weight, rshift_w).abs().long().clamp(0, 7).unsqueeze(0)
        products = hub.mapcbsg[input_index, weight_index] * torch.sign(weight).unsqueeze(0)
        output = torch.sign(input).unsqueeze(1) @ products.transpose(1, 2)
        return pow2_rshift(output, rshift_o).squeeze(1)

    assert torch.equal(hub(input_hub), _binary_conv_reference(hub, input_hub, hub_reference))


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
            with timer(device) as ref_elapsed:
                ref = F.conv2d(x, weight, bias, stride=1, padding=pad)
            for name, build in builders.items():
                module = build(pad)
                with timer(device) as elapsed:
                    y = module(x)
                rmse = (y - ref).pow(2).mean().sqrt().item()
                print(
                    f'[{device}] pad={pad} conv_{name}: rmse={rmse:.4f}, '
                    f'ratio={ref_elapsed.seconds / max(elapsed.seconds, 1e-12):.2f}x'
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


def make_performance_values():
    return (make_inputs()[0].repeat(4096, 1, 1, 1),)


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
    'make_performance_values': make_performance_values,
    'known_answer_case': known_answer_case,
    'gradient_case': gradient_case,
    'expected_ste_gradients': expected_ste_gradients,
    'extra_checks': _kernel_specific_checks,
}


def test_conv_binary():
    """Verify conv_binary quantization and STE gradients against its reference, including timing."""
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_conv_binary()
    test_conv_unarysim_quantization_semantics()

import torch
import torch.nn.functional as F

from napl.sim.module import conv_fxp
from napl.sim.module._shared import conv2d_output_shape, conv2d_get_padding, rshift_offset
from napl.utils import pow2_lshift, pow2_rshift
from napl.utils._shared_test import devices, non_streaming_suite, timer


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


def test_rshift_offset():
    """Verify dynamic fixed-point offsets reconstruct inputs and remain finite for zero operands."""
    for device in devices():
        inp = torch.randn(64, device=device) * 10
        wgt = torch.randn(32, 64, device=device) * 0.1
        ri, _rw, _ro = rshift_offset(inp, wgt, 7, 7, 'round', 1, 1)
        recon = pow2_lshift(pow2_rshift(inp, ri), ri)
        assert torch.allclose(recon, inp, atol=1e-4)

        zi, zw, zo = rshift_offset(
            torch.zeros(64, device=device), wgt, 7, 7, 'round', 1, 1
        )
        assert torch.isfinite(
            torch.tensor([float(zi), float(zw), float(zo)], device=device)
        ).all(), (device, zi, zw, zo)


def _kernel_specific_checks():
    """
    conv_fxp matches nn.Conv2d within its quantization bound for padding 0 and 1,
    and the STE lets gradients flow to input and weight.
    """
    torch.manual_seed(0)
    b, ic, oc, hw, k = 4, 3, 6, 10, 3
    x_cpu = torch.rand(b, ic, hw, hw) * 2 - 1
    weight_cpu = torch.rand(oc, ic, k, k) * 2 - 1
    bias_cpu = torch.rand(oc) * 2 - 1

    assert conv2d_get_padding((32, 32), (32, 32), 3, 1) == ((1, 1), (1, 1))

    for device in devices():
        x = x_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        for ksize, stride, pad in [(3, 1, 1), (5, 2, 0), (3, 2, 1)]:
            conv = torch.nn.Conv2d(
                2, 4, ksize, stride=stride, padding=pad
            ).to(device)
            out = conv(torch.zeros(1, 2, 32, 32, device=device))
            assert conv2d_output_shape(
                (32, 32), ksize, stride, pad
            ) == (out.shape[2], out.shape[3]), (device, ksize, stride, pad)
        builders = {
            'fxp': lambda pad: conv_fxp(ic, oc, k, padding=pad, weight_ext=weight, bias_ext=bias).to(device),
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


def make_random_perf_values():
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
    'gradient_atol': 1e-6,
    'gradient_rtol': 1e-6,
    'make_module_pair': make_module_pair,
    'make_inputs': make_inputs,
    'make_random_perf_values': make_random_perf_values,
    'known_answer_case': known_answer_case,
    'gradient_case': gradient_case,
    'expected_ste_gradients': expected_ste_gradients,
    'extra_checks': _kernel_specific_checks,
}


def test_conv_fxp():
    """Verify conv_fxp quantization and STE gradients against its reference, including timing."""
    non_streaming_suite(CONFIG)


if __name__ == '__main__':
    test_conv_fxp()
    test_conv_unarysim_quantization_semantics()
    test_rshift_offset()

import torch
import torch.nn.functional as F

from napl.sim.module import linear_hub
from napl.utils import pow2_rshift, rshift_offset
from napl.utils._shared_test import devices, single_shot_suite, timer


def test_linear_hub_truncates_scaled_magnitudes():
    """Verify linear_hub truncates scaled operands before accumulation like UnarySim."""
    input = torch.linspace(-0.49, 0.47, 18).reshape(3, 6)
    weight = torch.linspace(-0.46, 0.49, 24).reshape(4, 6)
    bias = torch.tensor([0.025, -0.04, 0.075, -0.09])
    module = linear_hub(
        6, 4, weight_ext=weight, bias_ext=bias,
        config={'widthi': 4, 'rngi': 'sobol', 'quantilei': 1,
                'widthw': 4, 'rngw': 'sobol', 'quantilew': 1,
                'cycle': 8, 'rounding': 'round'},
    )
    rshift_i, rshift_w, rshift_o = rshift_offset(input, weight, 3, 3, 'round')
    input_index = pow2_rshift(input, rshift_i).abs().long().clamp(0, 7).unsqueeze(1)
    weight_index = pow2_rshift(weight, rshift_w).abs().long().clamp(0, 7).unsqueeze(0)
    products = module.mapcbsg[input_index, weight_index] * torch.sign(weight).unsqueeze(0)
    reference = torch.sign(input).unsqueeze(1) @ products.transpose(1, 2)
    reference = pow2_rshift(reference, rshift_o).squeeze(1) + bias
    assert torch.equal(module(input), reference)


def test_linear_hub_short_cycle_shortens_run():
    """Verify a cycle below cycle_max sets the magnitude bitwidth like UnarySim HUBLinear."""
    torch.manual_seed(0)
    input_cpu = torch.rand(6, 12) * 2 - 1
    weight_cpu = torch.rand(5, 12) * 2 - 1
    bias_cpu = torch.rand(5) * 2 - 1
    cycle = 32

    for device in devices():
        input = input_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        module = linear_hub(
            12, 5, weight_ext=weight, bias_ext=bias,
            config={'widthi': 8, 'widthw': 8, 'cycle': cycle},
        ).to(device)
        # UnarySim: bitwidth = (cycle - 1).bit_length(), and the value map spans cycle levels.
        assert module.cycle_max == 128
        assert module.cycle_act == cycle
        assert module.width_act == 5
        assert tuple(module.mapcbsg.shape) == (cycle, cycle)

        rshift_i, rshift_w, rshift_o = rshift_offset(input, weight, 5, 5, 'round')
        input_index = pow2_rshift(input, rshift_i).abs().long().clamp(0, cycle - 1).unsqueeze(1)
        weight_index = pow2_rshift(weight, rshift_w).abs().long().clamp(0, cycle - 1).unsqueeze(0)
        products = module.mapcbsg[input_index, weight_index] * torch.sign(weight).unsqueeze(0)
        reference = torch.sign(input).unsqueeze(1) @ products.transpose(1, 2)
        reference = pow2_rshift(reference, rshift_o).squeeze(1) + bias
        assert torch.equal(module(input), reference)

        # The run length alone sets the resolution, so a width whose cap already equals
        # cycle gives the same result instead of the shorter run clipping magnitudes.
        matched = linear_hub(
            12, 5, weight_ext=weight, bias_ext=bias,
            config={'widthi': 6, 'widthw': 6, 'cycle': cycle},
        ).to(device)
        assert matched.cycle_max == cycle
        assert torch.equal(module(input), matched(input))
    print('Test passed.')


def test_linear_hub_non_power_of_two_cycle():
    """Verify a non-power-of-two cycle keeps exactly cycle levels, extending UnarySim HUBLinear."""
    torch.manual_seed(0)
    input_cpu = torch.rand(6, 12) * 2 - 1
    weight_cpu = torch.rand(5, 12) * 2 - 1
    bias_cpu = torch.rand(5) * 2 - 1
    cycle = 100

    for device in devices():
        input = input_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        module = linear_hub(
            12, 5, weight_ext=weight, bias_ext=bias,
            config={'widthi': 8, 'widthw': 8, 'cycle': cycle},
        ).to(device)
        # Upstream HUBLinear cannot be built here at all: its map spans the next power of
        # two and broadcasts against cycle. napl sizes the RNG sequences from the bitwidth
        # and spans exactly cycle levels.
        assert module.width_act == 7
        assert module.cycle_act == cycle
        assert tuple(module.mapcbsg.shape) == (cycle, cycle)

        rshift_i, rshift_w, rshift_o = rshift_offset(input, weight, 7, 7, 'round')
        input_index = pow2_rshift(input, rshift_i).abs().long().clamp(0, cycle - 1).unsqueeze(1)
        weight_index = pow2_rshift(weight, rshift_w).abs().long().clamp(0, cycle - 1).unsqueeze(0)
        products = module.mapcbsg[input_index, weight_index] * torch.sign(weight).unsqueeze(0)
        reference = torch.sign(input).unsqueeze(1) @ products.transpose(1, 2)
        reference = pow2_rshift(reference, rshift_o).squeeze(1) + bias
        assert torch.equal(module(input), reference)

        # The extra levels of the next power of two change the result, so the run is
        # not rounded up to it.
        rounded = linear_hub(
            12, 5, weight_ext=weight, bias_ext=bias,
            config={'widthi': 8, 'widthw': 8, 'cycle': 128},
        ).to(device)
        assert not torch.equal(module(input), rounded(input))
    print('Test passed.')


def test_linear_hub_cycle_validation():
    """Verify cycle takes int-valued floats and rejects fractional or non-positive values."""
    torch.manual_seed(0)
    input_cpu = torch.rand(6, 12) * 2 - 1
    weight_cpu = torch.rand(5, 12) * 2 - 1
    bias_cpu = torch.rand(5) * 2 - 1

    for device in devices():
        input = input_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        as_float = linear_hub(
            12, 5, weight_ext=weight, bias_ext=bias,
            config={'widthi': 8, 'widthw': 8, 'cycle': 32.0},
        ).to(device)
        as_int = linear_hub(
            12, 5, weight_ext=weight, bias_ext=bias,
            config={'widthi': 8, 'widthw': 8, 'cycle': 32},
        ).to(device)
        assert as_float.cycle_act == 32
        assert as_float.width_act == 5
        assert torch.equal(as_float(input), as_int(input))

        # bool is an int subclass and a string survives no arithmetic, so both are
        # rejected as cycle values, and the check runs before the cycle_max cap.
        for cycle in (32.5, 0, -4, True, '32', 1000.5):
            try:
                linear_hub(12, 5, weight_ext=weight, bias_ext=bias,
                           config={'widthi': 8, 'widthw': 8, 'cycle': cycle})
            except AssertionError as error:
                assert 'Invalid cycle' in str(error), (cycle, error)
                continue
            raise AssertionError(f'linear_hub accepted cycle {cycle}')

        # A bad widthi is reported against widthi, not against the cycle it derives.
        for widthi in (0, True, '8'):
            try:
                linear_hub(12, 5, weight_ext=weight, bias_ext=bias,
                           config={'widthi': widthi, 'widthw': widthi, 'cycle': 32})
            except AssertionError as error:
                assert 'Invalid widthi' in str(error), (widthi, error)
                continue
            raise AssertionError(f'linear_hub accepted widthi {widthi}')
    print('Test passed.')


def _kernel_specific_checks():
    """
    Binary-domain HUB linear (unary-multiplication value map) matches nn.Linear within
    the unary approximation bound, and the STE lets gradients flow.
    """
    torch.manual_seed(0)
    in_features, out_features, batch = 32, 16, 8
    weight_cpu = torch.rand(out_features, in_features) * 2 - 1
    bias_cpu = torch.rand(out_features) * 2 - 1
    x_cpu = torch.rand(batch, in_features) * 2 - 1

    for device in devices():
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        x = x_cpu.to(device)
        lin = linear_hub(
            in_features,
            out_features,
            bias=True,
            weight_ext=weight,
            bias_ext=bias,
        ).to(device)
        with timer(device) as elapsed:
            y = lin(x)
        with timer(device) as ref_elapsed:
            ref = F.linear(x, weight, bias)
        rmse = (y - ref).pow(2).mean().sqrt().item()
        print(
            f'[{device}] linear_hub rmse={rmse:.5f}, '
            f'max={(y-ref).abs().max().item():.5f}, '
            f'ratio={ref_elapsed.seconds / max(elapsed.seconds, 1e-12):.2f}x'
        )
        assert y.shape == ref.shape
        assert rmse < 0.06, (device, rmse)

        xg = x.clone().requires_grad_(True)
        lin(xg).sum().backward()
        assert xg.grad is not None and torch.isfinite(xg.grad).all()
        assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()

    print('Test passed.')


class linear_reference(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.register_buffer('weight', weight.detach().clone())
        self.register_buffer('bias', bias.detach().clone())


    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


def _make_candidate():
    weight = torch.tensor([
        [-0.5, -0.25, 0.25, 0.5],
        [0.5, 0.25, -0.25, -0.5],
    ])
    bias = torch.tensor([0.25, -0.25])
    candidate = linear_hub(
        4, 2, bias=True, weight_ext=weight, bias_ext=bias
    )
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    return candidate


def make_module_pair():
    candidate = _make_candidate()
    return candidate, linear_reference(candidate.weight, candidate.bias)


def make_inputs():
    return (torch.linspace(-0.75, 0.75, 32).reshape(8, 4),)


def make_performance_values():
    return (make_inputs()[0].repeat(4096, 1),)


def known_answer_case():
    candidate, reference = make_module_pair()
    values = torch.tensor([[0.5, 0.25, -0.25, -0.5]])
    return candidate, (values,), reference(values)


def gradient_case():
    return _make_candidate(), (torch.tensor([[0.5, 0.25, -0.25, -0.5]]),)


def expected_ste_gradients(candidate, _inputs, grad_output):
    return (grad_output @ candidate.weight.detach(),), {}


CONFIG = {
    'quantization_atol': 0.06,
    'known_answer_atol': 0.06,
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


def test_linear_hub():
    """Verify linear_hub quantization and STE gradients against its reference, including timing."""
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_linear_hub()
    test_linear_hub_truncates_scaled_magnitudes()
    test_linear_hub_short_cycle_shortens_run()
    test_linear_hub_non_power_of_two_cycle()
    test_linear_hub_cycle_validation()

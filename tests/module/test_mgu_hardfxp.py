import time

import torch
import torch.nn.functional as F

from napl.sim.module import mgu_hard, mgu_hardfxp
from napl.utils._shared_test import devices, single_shot_suite, sync


def _ref_mgu(x, hx, Wf, bf, Wn, bn):
    """Direct hard-activation MGU, independent of the cell implementation."""
    fg_in = F.hardtanh(F.linear(torch.cat((hx, x), 1), Wf, bf))
    fg = F.hardsigmoid(fg_in * 3)
    fg_hx = fg * hx
    ng = F.hardtanh(F.linear(torch.cat((fg_hx, x), 1), Wn, bn))
    return F.hardtanh(ng - fg * ng + fg_hx)


def _kernel_specific_checks():
    """
    mgu_hardfxp approximates mgu_hard within the fixed-point bound and trains with finite gradients.
    """
    torch.manual_seed(0)
    isz, hsz, b = 6, 4, 5
    cell = mgu_hard(isz, hsz, bias=True)
    x_cpu = torch.rand(b, isz) * 2 - 1
    hx_cpu = torch.rand(b, hsz) * 2 - 1

    for device in devices():
        cell = cell.to(device)
        x = x_cpu.to(device)
        hx = hx_cpu.to(device)
        sync(device)
        start = time.perf_counter()
        y = cell(x, hx)
        sync(device)
        elapsed = time.perf_counter() - start
        sync(device)
        start = time.perf_counter()
        _ref_mgu(
            x, hx, cell.weight_f, cell.bias_f, cell.weight_n, cell.bias_n
        )
        sync(device)
        ref_elapsed = time.perf_counter() - start

        cfx = mgu_hardfxp(
            isz, hsz, bias=True, config={'intwidth': 3, 'fracwidth': 6}
        ).to(device)
        for target, source in [
            (cfx.weight_f, cell.weight_f),
            (cfx.weight_n, cell.weight_n),
            (cfx.bias_f, cell.bias_f),
            (cfx.bias_n, cell.bias_n),
        ]:
            target.data = source.data.clone()
        rmse = (cfx(x, hx) - y).pow(2).mean().sqrt().item()
        print(
            f'[{device}] mgu_hardfxp rmse={rmse:.4f}, '
            f'hard/reference ratio={ref_elapsed / max(elapsed, 1e-12):.2f}x'
        )
        assert rmse < 0.05, (device, rmse)

        xg = x.clone().requires_grad_(True)
        cfx(xg, hx).sum().backward()
        assert torch.isfinite(xg.grad).all()
        assert torch.isfinite(cfx.weight_f.grad).all()
        cfx.zero_grad()

    print('Test passed.')


def _copy_parameters(target, source):
    for name in ('weight_f', 'bias_f', 'weight_n', 'bias_n'):
        getattr(target, name).data.copy_(getattr(source, name).data)


def _make_candidate():
    torch.manual_seed(17)
    candidate = mgu_hardfxp(
        6, 4, bias=True, config={'intwidth': 3, 'fracwidth': 6}
    )
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    return candidate


def make_module_pair():
    candidate = _make_candidate()
    reference = mgu_hard(6, 4, bias=True)
    _copy_parameters(reference, candidate)
    return candidate, reference


def make_inputs():
    return (
        torch.tensor([
            [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
            [0.75, 0.5, 0.25, 0.0, -0.25, -0.5],
        ]),
        torch.tensor([
            [-0.5, -0.25, 0.25, 0.5],
            [0.5, 0.25, -0.25, -0.5],
        ]),
    )


def known_answer_case():
    candidate, reference = make_module_pair()
    inputs = make_inputs()
    return candidate, inputs, reference(*inputs)


def gradient_case():
    return _make_candidate(), make_inputs()


def expected_ste_gradients(candidate, inputs, grad_output):
    reference = mgu_hardfxp(
        6, 4, bias=True, config={'intwidth': 3, 'fracwidth': 6}
    ).to(inputs[0].device)
    _copy_parameters(reference, candidate)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    refs = tuple(
        value.detach().clone().requires_grad_(True) for value in inputs
    )
    gradients = torch.autograd.grad(reference(*refs), refs, grad_output)
    return gradients, {}


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


def test_mgu_hardfxp():
    """Verify mgu_hardfxp quantization and STE gradients against its reference, including timing."""
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_mgu_hardfxp()

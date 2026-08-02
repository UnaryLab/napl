import time

import torch
import torch.nn.functional as F

from napl.sim.module import mgu_hard
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
    mgu_hard reproduces the hard-activation MGU equations exactly and trains with finite gradients.
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
        ref = _ref_mgu(
            x, hx, cell.weight_f, cell.bias_f, cell.weight_n, cell.bias_n
        )
        sync(device)
        ref_elapsed = time.perf_counter() - start
        assert y.shape == (b, hsz)
        assert torch.allclose(y, ref, atol=1e-5)
        assert cell(x).shape == (b, hsz)

        print(
            f'[{device}] hard/reference ratio='
            f'{ref_elapsed / max(elapsed, 1e-12):.2f}x'
        )

        xg = x.clone().requires_grad_(True)
        cell(xg, hx).sum().backward()
        assert torch.isfinite(xg.grad).all()
        assert torch.isfinite(cell.weight_f.grad).all()
        cell.zero_grad()

    print('Test passed.')


class mgu_reference(torch.nn.Module):
    def __init__(self, candidate):
        super().__init__()
        for name in ('weight_f', 'bias_f', 'weight_n', 'bias_n'):
            self.register_buffer(name, getattr(candidate, name).detach().clone())


    def forward(self, input, hx):
        return _ref_mgu(
            input, hx,
            self.weight_f, self.bias_f, self.weight_n, self.bias_n,
        )


def _make_candidate():
    torch.manual_seed(17)
    candidate = mgu_hard(6, 4, bias=True)
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    return candidate


def make_module_pair():
    candidate = _make_candidate()
    return candidate, mgu_reference(candidate)


def make_inputs():
    return (
        torch.linspace(-0.75, 0.75, 12).reshape(2, 6),
        torch.linspace(-0.5, 0.5, 8).reshape(2, 4),
    )


def known_answer_case():
    candidate, reference = make_module_pair()
    inputs = make_inputs()
    return candidate, inputs, reference(*inputs)


def gradient_case():
    return _make_candidate(), make_inputs()


def expected_ste_gradients(candidate, inputs, grad_output):
    refs = tuple(
        value.detach().clone().requires_grad_(True) for value in inputs
    )
    output = _ref_mgu(
        *refs,
        candidate.weight_f.detach(),
        candidate.bias_f.detach(),
        candidate.weight_n.detach(),
        candidate.bias_n.detach(),
    )
    gradients = torch.autograd.grad(output, refs, grad_output)
    return gradients, {}


CONFIG = {
    'quantization_atol': 1e-5,
    'known_answer_atol': 1e-5,
    'gradient_atol': 1e-6,
    'gradient_rtol': 1e-6,
    'make_module_pair': make_module_pair,
    'make_inputs': make_inputs,
    'known_answer_case': known_answer_case,
    'gradient_case': gradient_case,
    'expected_ste_gradients': expected_ste_gradients,
    'extra_checks': _kernel_specific_checks,
}


def test_mgu_hard():
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_mgu_hard()

import time

import torch
import torch.nn.functional as F

from napl.utils._shared_test import devices, single_shot_suite, sync
from napl.sim.module.mgu_hardpt import mgu_hardpt


def _ref_mgu_pt(x, hx, Wih, bih, Whh, bhh):
    """Direct hard-activation PT-style MGU, independent implementation."""
    gi = F.hardtanh(F.linear(x, Wih, bih))
    gh = F.hardtanh(F.linear(hx, Whh, bhh))
    i_f, i_n = gi.chunk(2, 1)
    h_f, h_n = gh.chunk(2, 1)
    fg = F.hardsigmoid(F.hardtanh(i_f + h_f) * 3)
    ng = F.hardtanh(i_n + fg * h_n)
    return F.hardtanh(ng - fg * ng + fg * hx)


def _kernel_specific_checks():
    """
    Correctness: mgu_hardpt reproduces the PT-style hard-activation MGU equations
    exactly on every device; hx=None default; trainable gradients.
    Performance: cell vs the functional reference on identical inputs, per device.
    """
    torch.manual_seed(0)
    isz, hsz, b = 6, 4, 5
    cell = mgu_hardpt(isz, hsz, bias=True)
    x0 = torch.rand(b, isz) * 2 - 1
    hx0 = torch.rand(b, hsz) * 2 - 1

    for device in devices():
        c = cell.to(device)
        x, hx = x0.to(device), hx0.to(device)
        y = c(x, hx)
        ref = _ref_mgu_pt(x, hx, c.weight_ih, c.bias_ih, c.weight_hh, c.bias_hh)
        assert y.shape == (b, hsz)
        assert torch.allclose(y, ref, atol=1e-6), f'mismatch on {device}'
        assert c(x).shape == (b, hsz)  # hx=None default
        # output stays in the legal unary range
        assert y.abs().max().item() <= 1.0 + 1e-6

        # gradients flow (hard activations are piecewise-linear, autograd-native)
        xg = x.clone().requires_grad_(True)
        c(xg, hx).sum().backward()
        assert torch.isfinite(xg.grad).all()
        assert torch.isfinite(c.weight_ih.grad).all()
        c.zero_grad()

        # performance: cell vs functional reference, identical inputs
        n = 50
        sync(device)
        t0 = time.perf_counter()
        for _ in range(n):
            c(x, hx)
        sync(device)
        t_cell = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n):
            _ref_mgu_pt(x, hx, c.weight_ih, c.bias_ih, c.weight_hh, c.bias_hh)
        sync(device)
        t_ref = time.perf_counter() - t0
        print(f'[{device}] mgu_hardpt {t_cell*1e3/n:.3f} ms/iter, '
              f'reference {t_ref*1e3/n:.3f} ms/iter, ratio {t_ref/max(t_cell,1e-12):.2f}x')

    # soft (hard=False) path uses true sigmoid/tanh
    soft = mgu_hardpt(isz, hsz, bias=True, config={'hard': False})
    assert soft(x0, hx0).shape == (b, hsz)
    print('PASS')


class mgu_reference(torch.nn.Module):
    def __init__(self, candidate):
        super().__init__()
        for name in ('weight_ih', 'bias_ih', 'weight_hh', 'bias_hh'):
            self.register_buffer(name, getattr(candidate, name).detach().clone())

    def forward(self, input, hx):
        return _ref_mgu_pt(
            input, hx,
            self.weight_ih, self.bias_ih, self.weight_hh, self.bias_hh,
        )


def _make_candidate():
    torch.manual_seed(17)
    candidate = mgu_hardpt(6, 4, bias=True)
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
    output = _ref_mgu_pt(
        *refs,
        candidate.weight_ih.detach(),
        candidate.bias_ih.detach(),
        candidate.weight_hh.detach(),
        candidate.bias_hh.detach(),
    )
    gradients = torch.autograd.grad(output, refs, grad_output)
    return gradients, {}


CONFIG = {
    'quantization_atol': 1e-6,
    'known_answer_atol': 1e-6,
    'gradient_atol': 1e-6,
    'gradient_rtol': 1e-6,
    'make_module_pair': make_module_pair,
    'make_inputs': make_inputs,
    'known_answer_case': known_answer_case,
    'gradient_case': gradient_case,
    'expected_ste_gradients': expected_ste_gradients,
    'extra_checks': _kernel_specific_checks,
}


def test_mgu_hardpt():
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_mgu_hardpt()

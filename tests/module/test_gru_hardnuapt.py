import time

import torch
import torch.nn.functional as F

from napl.utils._shared_test import devices, single_shot_suite, sync
from napl.sim.module.gru_hardnuapt import gru_hardnuapt


def _ref_gru_hard(x, hx, w_ih, w_hh, b_ih, b_hh):
    """Direct hard-activation GRU (PyTorch GRUCell equations), independent implementation."""
    i_r, i_z, i_n = F.linear(x, w_ih, b_ih).chunk(3, 1)
    h_r, h_z, h_n = F.linear(hx, w_hh, b_hh).chunk(3, 1)
    rg = F.hardsigmoid((i_r + h_r) * 3)
    ug = F.hardsigmoid((i_z + h_z) * 3)
    ng = F.hardtanh(i_n + rg * h_n)
    return (1 - ug) * ng + ug * hx


def _kernel_specific_checks():
    """
    Correctness: hard=True matches the hard-activation GRU equations exactly;
    hard=False matches nn.GRUCell exactly (same weights). Gradients flow.
    Performance: timed against nn.GRUCell on every device.
    """
    torch.manual_seed(0)
    isz, hsz, b = 6, 4, 5
    for device in devices():
        cell = gru_hardnuapt(isz, hsz, bias=True).to(device)
        x = (torch.rand(b, isz, device=device) * 2 - 1)
        hx = (torch.rand(b, hsz, device=device) * 2 - 1)

        # Check hard activations against an independent reference.
        y = cell(x, hx)
        ref = _ref_gru_hard(x, hx, cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh)
        assert y.shape == (b, hsz)
        assert torch.allclose(y, ref, atol=1e-6), device
        assert cell(x).shape == (b, hsz)  # Default hx=None path.

        # Check soft activations against nn.GRUCell with identical weights.
        soft = gru_hardnuapt(isz, hsz, bias=True, config={'hard': False}).to(device)
        gru = torch.nn.GRUCell(isz, hsz, bias=True).to(device)
        with torch.no_grad():
            gru.weight_ih.copy_(soft.weight_ih)
            gru.weight_hh.copy_(soft.weight_hh)
            gru.bias_ih.copy_(soft.bias_ih)
            gru.bias_hh.copy_(soft.bias_hh)
        assert torch.allclose(soft(x, hx), gru(x, hx), atol=1e-5), device

        xg = x.clone().requires_grad_(True)
        cell(xg, hx).sum().backward()
        assert torch.isfinite(xg.grad).all()
        assert torch.isfinite(cell.weight_ih.grad).all()

        # Compare against nn.GRUCell on identical inputs.
        xb = torch.rand(256, isz, device=device) * 2 - 1
        hb = torch.rand(256, hsz, device=device) * 2 - 1
        for _ in range(3):  # Warm up before timing.
            cell(xb, hb)
            gru(xb, hb)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(50):
            cell(xb, hb)
        sync(device)
        t1 = time.perf_counter()
        for _ in range(50):
            gru(xb, hb)
        sync(device)
        t2 = time.perf_counter()
        print(f'[{device}] gru_hardnuapt {t1 - t0:.4f}s vs nn.GRUCell {t2 - t1:.4f}s '
              f'(ratio {(t1 - t0) / max(t2 - t1, 1e-9):.2f}x)')

    print('Test passed.')


class gru_reference(torch.nn.Module):
    def __init__(self, candidate):
        super().__init__()
        for name in ('weight_ih', 'bias_ih', 'weight_hh', 'bias_hh'):
            self.register_buffer(name, getattr(candidate, name).detach().clone())

    def forward(self, input, hx):
        return _ref_gru_hard(
            input, hx,
            self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh,
        )


def _make_candidate():
    torch.manual_seed(17)
    candidate = gru_hardnuapt(6, 4, bias=True)
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)
    return candidate


def make_module_pair():
    candidate = _make_candidate()
    return candidate, gru_reference(candidate)


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
    output = _ref_gru_hard(
        *refs,
        candidate.weight_ih.detach(),
        candidate.weight_hh.detach(),
        candidate.bias_ih.detach(),
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


def test_gru_hardnuapt():
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_gru_hardnuapt()

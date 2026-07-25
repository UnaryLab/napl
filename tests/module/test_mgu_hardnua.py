import sys
import time

import torch
import torch.nn.functional as F

from napl.utils._shared_test import devices, single_shot_suite, sync
from napl.sim.module.mgu_hardnua import mgu_hardnua

sys.path.insert(0, '/Users/diwu/Projects')
from UnarySim.kernel.rnn import HardMGUCellNUA


def _ref_mgu_nua(x, hx, Wf, bf, Wn, bn):
    """Direct NUA hard MGU, independent of both implementations (no hardtanh clamps)."""
    fg = F.hardsigmoid(F.linear(torch.cat((hx, x), 1), Wf, bf) * 3)
    fg_hx = fg * hx
    ng = F.hardtanh(F.linear(torch.cat((fg_hx, x), 1), Wn, bn))
    return ng - fg * ng + fg_hx


def _kernel_specific_checks():
    """
    mgu_hardnua reproduces the NUA hard MGU equations exactly, matches UnarySim
    HardMGUCellNUA bit-exactly on identical inputs/weights, and trains, on every device.
    """
    torch.manual_seed(0)
    isz, hsz, b = 6, 4, 5
    cell = mgu_hardnua(isz, hsz, bias=True)
    ref_cell = HardMGUCellNUA(isz, hsz, bias=True, hard=True)
    for dst, src in [(ref_cell.weight_f, cell.weight_f), (ref_cell.weight_n, cell.weight_n),
                     (ref_cell.bias_f, cell.bias_f), (ref_cell.bias_n, cell.bias_n)]:
        dst.data = src.data.clone()
    x0 = torch.rand(b, isz) * 2 - 1
    hx0 = torch.rand(b, hsz) * 2 - 1

    for device in devices():
        cell.to(device)
        ref_cell.to(device)
        x, hx = x0.to(device), hx0.to(device)

        y = cell(x, hx)
        assert y.shape == (b, hsz)
        # analytic reference
        ref = _ref_mgu_nua(x, hx, cell.weight_f, cell.bias_f, cell.weight_n, cell.bias_n)
        assert torch.allclose(y, ref, atol=1e-6), (device, (y - ref).abs().max().item())
        # UnarySim faithfulness: identical inputs and weights -> bit-exact
        y_us = ref_cell(x, hx)
        assert torch.equal(y, y_us), (device, (y - y_us).abs().max().item())
        # NUA property: output may leave [-1, 1] (unlike mgu_hard); just check finite
        assert torch.isfinite(y).all()
        assert cell(x).shape == (b, hsz)   # hx=None default

        # gradients flow
        xg = x.clone().requires_grad_(True)
        cell(xg, hx).sum().backward()
        assert torch.isfinite(xg.grad).all()
        assert torch.isfinite(cell.weight_f.grad).all()
        cell.zero_grad()

        # performance vs the UnarySim baseline, identical inputs
        n = 200
        for m in (cell, ref_cell):
            m(x, hx)   # warmup
        sync(device)
        t0 = time.perf_counter()
        for _ in range(n):
            cell(x, hx)
        sync(device)
        t1 = time.perf_counter()
        for _ in range(n):
            ref_cell(x, hx)
        sync(device)
        t2 = time.perf_counter()
        print(f'[{device}] napl {t1 - t0:.4f}s vs UnarySim {t2 - t1:.4f}s '
              f'-> speedup {(t2 - t1) / max(t1 - t0, 1e-12):.2f}x')

    print('Test passed.')


class mgu_reference(torch.nn.Module):
    def __init__(self, candidate):
        super().__init__()
        for name in ('weight_f', 'bias_f', 'weight_n', 'bias_n'):
            self.register_buffer(name, getattr(candidate, name).detach().clone())

    def forward(self, input, hx):
        return _ref_mgu_nua(
            input, hx,
            self.weight_f, self.bias_f, self.weight_n, self.bias_n,
        )


def _make_candidate():
    torch.manual_seed(17)
    candidate = mgu_hardnua(6, 4, bias=True)
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
    output = _ref_mgu_nua(
        *refs,
        candidate.weight_f.detach(),
        candidate.bias_f.detach(),
        candidate.weight_n.detach(),
        candidate.bias_n.detach(),
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


def test_mgu_hardnua():
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_mgu_hardnua()

"""Flux-stability profile for the mgu_hard_mix streaming module (single-step model)."""

import torch
import torch.nn.functional as F

from napl.sim.module import mgu_hard_mix

from profile_common import profile_op

ISZ, HSZ, BATCH = 4, 3, 2


def _gate_parameters():
    """Return the test's deterministic linspace gate tensors (weight_f, bias_f, weight_n, bias_n)."""
    weight_f = torch.linspace(-0.5, 0.5, HSZ * (HSZ + ISZ)).view(HSZ, HSZ + ISZ)
    weight_n = torch.linspace(0.5, -0.5, HSZ * (HSZ + ISZ)).view(HSZ, HSZ + ISZ)
    bias_f = torch.linspace(-0.2, 0.2, HSZ)
    bias_n = torch.linspace(0.2, -0.2, HSZ)
    return weight_f, bias_f, weight_n, bias_n


def _hard_mgu(input_value, hx_value):
    """One binary-domain hard-MGU step (the test's exact float math)."""
    weight_f, bias_f, weight_n, bias_n = _gate_parameters()
    fg_in = F.hardtanh(F.linear(torch.cat((hx_value, input_value), 1), weight_f, bias_f))
    fg = F.hardsigmoid(fg_in * 3)
    fg_hx = fg * hx_value
    ng = F.hardtanh(F.linear(torch.cat((fg_hx, input_value), 1), weight_n, bias_n))
    return F.hardtanh(ng - fg * ng + fg_hx)


def profile():
    """Return the flux-stability result dict for mgu_hard_mix (bipolar only, single-step model)."""
    weight_f, bias_f, weight_n, bias_n = _gate_parameters()
    hx_value = torch.linspace(-0.6, 0.6, BATCH * HSZ).view(BATCH, HSZ)
    config = {'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol'}
    make_op = lambda: mgu_hard_mix(weight_f, bias_f, weight_n, bias_n, hx_value, config)
    # Single-step model: hx is a fresh re-encoded fixed-value input stream, not an
    # output fed back, so both input and hx are monitored effective inputs.
    result = profile_op(
        None,
        make_op=make_op,
        ctor=config,
        inputs=[
            {'range': (-0.6, 0.6), 'polarity': 'bipolar', 'shape': (BATCH, ISZ), 'dim': 1},
            # hx streams the fixed hidden value the op multiplies by internally, so its
            # sample must equal the baked hx_value buffer for the reference to hold.
            {'range': (-0.6, 0.6), 'polarity': 'bipolar', 'shape': (BATCH, HSZ), 'dim': 2,
             'sample': lambda shape: hx_value},
        ],
        reference=lambda values, polarity: _hard_mgu(values[0], values[1]),
        apply=lambda op, spikes, values: op(spikes[0], spikes[1]),
        output_polarity='bipolar',
        timesteps=256,
        seed=0,
    )
    result['model'] = 'single_step'
    return [result]


if __name__ == '__main__':
    for r in profile():
        print(r)

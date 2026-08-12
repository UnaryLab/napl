"""Flux-stability profile for the conv_gaines streaming module.

Scaled Gaines addition requires a power-of-two fan-in, so this profile uses a
2x2 kernel over 4 input channels (entry = 16) rather than the 3x3x3 = 27 fan-in
of the uGEMM/mix conv profiles, which is not a power of two.
"""

import torch
import torch.nn.functional as F

from napl.sim.module import conv_gaines

from profile_common import profile_module


def profile():
    """Return the flux-stability result dicts for conv_gaines, one per supported polarity."""
    n, ic, oc, hw, k = 2, 4, 4, 8, 2
    entry = ic * k * k  # bias=None; power-of-two fan-in required by scaled Gaines addition = 16.
    results = []
    for polarity in ('unipolar', 'bipolar'):
        lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
        # Deterministic small weight keeps conv2d(a, w) / entry inside the legal range.
        weight = torch.linspace(-0.5, 0.5, oc * ic * k * k).view(oc, ic, k, k)
        if polarity == 'unipolar':
            weight = weight.abs()
        # Weight streams occupy Sobol dims 2..17; activation uses dim 1 to decorrelate.
        make_op = lambda w=weight, p=polarity: conv_gaines(
            w, None, stride=1, padding=0,
            config={'polarity': p, 'timestep': 256, 'generator': 'sobol', 'dim': 2, 'scaled': True},
        )
        results.append(profile_module(
            make_op,
            activation={'range': (lo, hi), 'polarity': polarity, 'shape': (n, ic, hw, hw),
                        'dim': 1, 'generator': 'sobol',
                        'unfold': {'kernel_size': k, 'stride': 1, 'padding': 0, 'fold_channels': True}},
            reference=lambda a, p, w=weight, e=entry: F.conv2d(a, w) / e,
            polarity=polarity,
        ))
    return results


if __name__ == '__main__':
    for r in profile():
        print(r)

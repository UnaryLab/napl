"""Flux-stability profile for the conv_mix streaming module."""

import torch
import torch.nn.functional as F

from napl.sim.module import conv_mix

from profile_common import profile_module


def profile():
    """Return the flux-stability result dicts for conv_mix, one per supported polarity."""
    n, ic, oc, hw, k = 2, 3, 4, 8, 3
    entry = ic * k * k  # bias=None; fan-in of one 3x3x3 patch = 27.
    results = []
    for polarity in ('unipolar', 'bipolar'):
        lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
        # Deterministic small weight keeps conv2d(a, w) / entry inside the legal range.
        weight = torch.linspace(-0.5, 0.5, oc * ic * k * k).view(oc, ic, k, k)
        if polarity == 'unipolar':
            weight = weight.abs()
        # Weight encoder uses dim 2 (bias dim 3, pad dim 4); activation uses dim 1 to decorrelate.
        make_op = lambda w=weight, p=polarity: conv_mix(
            w, None, stride=1, padding=0,
            config={'polarity': p, 'timestep': 256, 'generator': 'sobol', 'dim': 2, 'scale': None, 'width': 12},
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

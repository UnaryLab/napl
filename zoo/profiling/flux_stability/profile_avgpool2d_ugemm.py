"""Flux-stability profile for the avgpool2d_ugemm streaming module."""

import torch.nn.functional as F

from napl.sim.module import avgpool2d_ugemm

from profile_common import profile_module


def profile():
    """Return the flux-stability result dicts for avgpool2d_ugemm, one per supported polarity."""
    n, c, hw, k = 2, 3, 8, 2  # stride defaults to k, so windows do not overlap.
    results = []
    for polarity in ('unipolar', 'bipolar'):
        lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
        # No weight: the pool has only a kernel size and the reference is a true window mean.
        make_op = lambda p=polarity: avgpool2d_ugemm(k, config={'polarity': p})
        results.append(profile_module(
            make_op,
            activation={'range': (lo, hi), 'polarity': polarity, 'shape': (n, c, hw, hw),
                        'dim': 1, 'generator': 'sobol',
                        'unfold': {'kernel_size': k, 'stride': k, 'padding': 0, 'fold_channels': False}},
            reference=lambda a, p, ks=k: F.avg_pool2d(a, ks),
            polarity=polarity,
        ))
    return results


if __name__ == '__main__':
    for r in profile():
        print(r)

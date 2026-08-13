"""Flux-stability profile for the linear_gaines streaming module."""

import torch

from napl.sim.module import linear_gaines

from profile_common import profile_module


def profile():
    """Return the flux-stability result dicts for linear_gaines, one per supported polarity."""
    # Scaled mode needs a power-of-two entry; in_features=8 (bias=None) satisfies it.
    out_features, in_features = 4, 8
    results = []
    for polarity in ('unipolar', 'bipolar'):
        lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
        # Deterministic small weight keeps (a @ w.T) / entry inside the legal range.
        weight = torch.linspace(-0.5, 0.5, out_features * in_features).view(out_features, in_features)
        if polarity == 'unipolar':
            weight = weight.abs()
        entry = in_features  # bias=None; power of two, so _scaled_ref reduces to (a @ w.T) / entry.
        # Module weight sequences start at dim 2; activation uses dim 1 to decorrelate.
        make_op = lambda w=weight, p=polarity: linear_gaines(
            w, None,
            {'polarity': p, 'timestep': 256, 'generator': 'sobol', 'dim': 2, 'scaled': True},
        )
        results.append(profile_module(
            make_op,
            activation={'range': (lo, hi), 'polarity': polarity, 'shape': (32, in_features),
                        'dim': 1, 'generator': 'sobol', 'reduce_dim': -1},
            reference=lambda a, p, w=weight, e=entry: (a @ w.T) / e,
            polarity=polarity,
        ))
    return results


if __name__ == '__main__':
    for r in profile():
        print(r)

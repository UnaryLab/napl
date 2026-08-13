"""Flux-stability profile for the clamp streaming operation."""

import torch

from napl.sim.operation import clamp

from profile_common import profile_op

# Fixed band per polarity, both bounds inside the polarity's legal value range.
BOUNDS = {'bipolar': (-0.5, 0.5), 'unipolar': (0.25, 0.75)}


def profile():
    """Return the flux-stability result dicts for clamp, one per supported polarity."""
    results = []
    for polarity in ('bipolar', 'unipolar'):
        lo, hi = BOUNDS[polarity]
        rng = (-1.0, 1.0) if polarity == 'bipolar' else (0.0, 1.0)
        results.append(
            profile_op(
                clamp,
                ctor={'polarity': polarity, 'lo': lo, 'hi': hi},
                inputs=[{'range': rng, 'polarity': polarity, 'shape': (16384,)}],
                reference=lambda values, _polarity, lo=lo, hi=hi: torch.clamp(values[0], lo, hi),
                timesteps=256,
                seed=0,
            )
        )
    return results


if __name__ == '__main__':
    for r in profile():
        print(r)

"""Flux-stability profile for the and_corr streaming operation."""

import torch

from napl.sim.operation import and_corr

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for and_corr, one per supported polarity."""
    return [
        profile_op(
            and_corr,
            ctor={'polarity': 'unipolar'},
            inputs=[
                # Both operands on the SAME Sobol dim (dim 1) so they are
                # maximally correlated (SCC +1) and AND realizes the minimum;
                # profile_op would otherwise assign each input a distinct dim.
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 1},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 1},
            ],
            reference=lambda values, polarity: torch.minimum(values[0], values[1]),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

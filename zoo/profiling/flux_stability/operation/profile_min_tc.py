"""Flux-stability profile for the min_tc streaming operation."""

import torch

from napl.sim.operation import min_tc

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for min_tc, one per supported polarity."""
    return [
        profile_op(
            min_tc,
            ctor={'polarity': 'unipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'generator': 'temporal'},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'generator': 'temporal'},
            ],
            reference=lambda values, polarity: torch.minimum(values[0], values[1]),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            min_tc,
            ctor={'polarity': 'bipolar'},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,), 'generator': 'temporal'},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,), 'generator': 'temporal'},
            ],
            reference=lambda values, polarity: torch.minimum(values[0], values[1]),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

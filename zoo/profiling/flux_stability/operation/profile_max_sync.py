"""Flux-stability profile for the max_sync streaming operation."""

import torch

from napl.sim.operation import max_sync

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for max_sync, one per supported polarity."""
    return [
        profile_op(
            max_sync,
            ctor={'polarity': 'unipolar', 'depth': 1},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: torch.maximum(values[0], values[1]),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            max_sync,
            ctor={'polarity': 'bipolar', 'depth': 1},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: torch.maximum(values[0], values[1]),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

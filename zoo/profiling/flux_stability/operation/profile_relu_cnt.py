"""Flux-stability profile for the relu_cnt streaming operation."""

import torch

from napl.sim.operation import relu_cnt

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for relu_cnt, one per supported polarity."""
    return [
        profile_op(
            relu_cnt,
            ctor={'width': 3, 'polarity': 'bipolar'},
            inputs=[{'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: torch.relu(values[0]),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

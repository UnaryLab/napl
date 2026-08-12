"""Flux-stability profile for the relu_sat streaming operation."""

import torch

from napl.sim.operation import relu_sat

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for relu_sat, one per supported polarity."""
    return [
        profile_op(
            relu_sat,
            ctor={'polarity': 'bipolar'},
            inputs=[{'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: torch.relu(values[0]),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

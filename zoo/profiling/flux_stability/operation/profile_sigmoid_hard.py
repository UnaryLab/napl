"""Flux-stability profile for the sigmoid_hard streaming operation."""

import torch

from napl.sim.operation import sigmoid_hard

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for sigmoid_hard, one per supported polarity."""
    return [
        profile_op(
            sigmoid_hard,
            ctor={'polarity': 'unipolar'},
            inputs=[{'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: torch.nn.functional.hardsigmoid(values[0] * 3),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            sigmoid_hard,
            ctor={'polarity': 'bipolar'},
            inputs=[{'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: torch.nn.functional.hardsigmoid(values[0] * 3),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

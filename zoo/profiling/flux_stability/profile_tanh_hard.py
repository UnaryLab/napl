"""Flux-stability profile for the tanh_hard streaming operation."""

import torch

from napl.sim.operation import tanh_hard

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for tanh_hard, one per supported polarity."""
    return [
        profile_op(
            tanh_hard,
            ctor={'polarity': 'unipolar'},
            inputs=[{'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: torch.nn.functional.hardtanh(values[0]),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            tanh_hard,
            ctor={'polarity': 'bipolar'},
            inputs=[{'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: torch.nn.functional.hardtanh(values[0]),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

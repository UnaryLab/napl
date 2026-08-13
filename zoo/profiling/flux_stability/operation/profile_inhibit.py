"""Flux-stability profile for the inhibit streaming operation."""

import torch

from napl.sim.operation import inhibit

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for inhibit, one per supported polarity."""
    return [
        profile_op(
            inhibit,
            ctor={'polarity': 'unipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'generator': 'temporal'},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'generator': 'temporal'},
            ],
            # An inhibited data stream never falls, decoding to the ceiling value 1.0.
            reference=lambda values, polarity: torch.where(
                values[0] <= values[1], values[0], torch.full_like(values[0], 1.0)),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            inhibit,
            ctor={'polarity': 'bipolar'},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,), 'generator': 'temporal'},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,), 'generator': 'temporal'},
            ],
            # An inhibited data stream never falls, decoding to the ceiling value 1.0.
            reference=lambda values, polarity: torch.where(
                values[0] <= values[1], values[0], torch.full_like(values[0], 1.0)),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

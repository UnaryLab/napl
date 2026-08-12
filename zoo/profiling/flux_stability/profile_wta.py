"""Flux-stability profile for the wta streaming operation."""

import torch

from napl.sim.operation import wta

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for wta, one per supported polarity."""
    return [
        profile_op(
            wta,
            ctor={'polarity': 'unipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'generator': 'temporal'},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'generator': 'temporal'},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'generator': 'temporal'},
            ],
            # wta selects the smallest value; its input streams are stacked on a new last dim.
            reference=lambda values, polarity: torch.minimum(torch.minimum(values[0], values[1]), values[2]),
            apply=lambda op, spikes, values: op(torch.stack(spikes, dim=-1), dim=-1),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            wta,
            ctor={'polarity': 'bipolar'},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,), 'generator': 'temporal'},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,), 'generator': 'temporal'},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,), 'generator': 'temporal'},
            ],
            # wta selects the smallest value; its input streams are stacked on a new last dim.
            reference=lambda values, polarity: torch.minimum(torch.minimum(values[0], values[1]), values[2]),
            apply=lambda op, spikes, values: op(torch.stack(spikes, dim=-1), dim=-1),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

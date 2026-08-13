"""Flux-stability profile for the max_rc streaming operation."""

import torch

from napl.sim.operation import max_rc

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for max_rc, one per supported polarity."""
    return [
        profile_op(
            max_rc,
            ctor={'polarity': 'unipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            # Profile the value output (max_rc's second output is an argmax indicator).
            reference=lambda values, polarity: torch.maximum(values[0], values[1]),
            apply=lambda op, spikes, values: op(*spikes)[0],
            timesteps=256,
            seed=0,
        ),
        profile_op(
            max_rc,
            ctor={'polarity': 'bipolar'},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            # Profile the bipolar value output (max_rc's second output is a unipolar argmax).
            reference=lambda values, polarity: torch.maximum(values[0], values[1]),
            apply=lambda op, spikes, values: op(*spikes)[0],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

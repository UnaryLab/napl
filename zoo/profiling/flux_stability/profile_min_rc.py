"""Flux-stability profile for the min_rc streaming operation."""

import torch

from napl.sim.operation import min_rc

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for min_rc, one per supported polarity."""
    return [
        profile_op(
            min_rc,
            ctor={'polarity': 'unipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            # Profile the value output (min_rc's second output is an argmin indicator).
            reference=lambda values, polarity: torch.minimum(values[0], values[1]),
            apply=lambda op, spikes, values: op(*spikes)[0],
            timesteps=256,
            seed=0,
        ),
        profile_op(
            min_rc,
            ctor={'polarity': 'bipolar'},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            # Profile the bipolar value output (min_rc's second output is a unipolar argmin).
            reference=lambda values, polarity: torch.minimum(values[0], values[1]),
            apply=lambda op, spikes, values: op(*spikes)[0],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

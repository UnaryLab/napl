"""Flux-stability profile for the relu_tc streaming operation."""

import torch

from napl.sim.operation import relu_tc

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for relu_tc, one per supported polarity."""
    return [
        profile_op(
            relu_tc,
            ctor={'width': 8, 'polarity': 'bipolar'},
            # relu_tc is temporal-coded, so the input stream uses the temporal generator.
            inputs=[{'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,), 'generator': 'temporal'}],
            reference=lambda values, polarity: torch.relu(values[0]),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

"""Flux-stability profile for the tanh_pn streaming operation."""

import torch

from napl.sim.operation import tanh_pn

from profile_common import profile_op

# tanh_pn depth 3 scales its input by a gain of 2 ** (depth - 1).
_GAIN = 2 ** (3 - 1)


def profile():
    """Return the flux-stability result dicts for tanh_pn, one per supported polarity."""
    return [
        profile_op(
            tanh_pn,
            ctor={'depth': 3, 'polarity': 'bipolar'},
            inputs=[{'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: torch.tanh(values[0] * _GAIN),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

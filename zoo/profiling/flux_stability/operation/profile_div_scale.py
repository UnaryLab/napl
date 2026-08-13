"""Flux-stability profile for the div_scale streaming operation."""

from napl.sim.operation import div_scale

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for div_scale, one per supported polarity."""
    # A single encoded stream is divided by the fixed power-of-two scale of 2. Dividing a
    # full-range stream by 2 keeps the quotient in the legal range for both polarities.
    return [
        profile_op(
            div_scale,
            ctor={'polarity': 'unipolar', 'scale': 2, 'intwidth': 12, 'fracwidth': 4},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 1},
            ],
            reference=lambda values, polarity: values[0] / 2,
            timesteps=256,
            seed=0,
        ),
        profile_op(
            div_scale,
            ctor={'polarity': 'bipolar', 'scale': 2, 'intwidth': 12, 'fracwidth': 4},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,), 'dim': 1},
            ],
            reference=lambda values, polarity: values[0] / 2,
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

"""Flux-stability profile for the div_scale_dyn streaming operation."""

from napl.sim.operation import div_scale_dyn

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for div_scale_dyn, one per supported polarity."""
    # The per-call scale is held at the fixed value 2, matching the divisor of the reference.
    # Dividing a full-range stream by 2 keeps the quotient in the legal range for both polarities.
    return [
        profile_op(
            div_scale_dyn,
            ctor={'polarity': 'unipolar', 'scale_max': 4, 'intwidth': 12, 'fracwidth': 4},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 1},
            ],
            reference=lambda values, polarity: values[0] / 2,
            apply=lambda op, spikes, values: op(spikes[0], 2),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            div_scale_dyn,
            ctor={'polarity': 'bipolar', 'scale_max': 4, 'intwidth': 12, 'fracwidth': 4},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,), 'dim': 1},
            ],
            reference=lambda values, polarity: values[0] / 2,
            apply=lambda op, spikes, values: op(spikes[0], 2),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

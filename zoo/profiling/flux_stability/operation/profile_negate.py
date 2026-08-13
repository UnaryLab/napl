"""Flux-stability profile for the negate streaming operation."""

from napl.sim.operation import negate

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for negate, one per supported polarity."""
    return [
        profile_op(
            negate,
            ctor={'polarity': 'bipolar'},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: -values[0],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

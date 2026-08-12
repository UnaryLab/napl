"""Flux-stability profile for the shiftreg streaming operation."""

from napl.sim.operation import shiftreg

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for shiftreg, one per supported polarity."""
    # A fixed-depth shift register preserves the steady-state input value.
    return [
        profile_op(
            shiftreg,
            ctor={'depth': 2, 'polarity': 'unipolar'},
            inputs=[{'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: values[0],
            timesteps=256,
            seed=0,
        ),
        profile_op(
            shiftreg,
            ctor={'depth': 2, 'polarity': 'bipolar'},
            inputs=[{'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: values[0],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

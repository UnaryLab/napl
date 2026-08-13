"""Flux-stability profile for the dff streaming operation."""

from napl.sim.operation import dff

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for dff, one per supported polarity."""
    # A fixed delay preserves the steady-state input value.
    return [
        profile_op(
            dff,
            ctor={'depth': 3, 'polarity': 'unipolar'},
            inputs=[{'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: values[0],
            timesteps=256,
            seed=0,
        ),
        profile_op(
            dff,
            ctor={'depth': 3, 'polarity': 'bipolar'},
            inputs=[{'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: values[0],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

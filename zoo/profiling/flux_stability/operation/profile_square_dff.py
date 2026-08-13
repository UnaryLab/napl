"""Flux-stability profile for the square_dff streaming operation."""

from napl.sim.operation import square_dff

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for square_dff, one per supported polarity."""
    return [
        profile_op(
            square_dff,
            ctor={'polarity': 'unipolar', 'depth': 1},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0].square(),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            square_dff,
            ctor={'polarity': 'bipolar', 'depth': 1},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0].square(),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

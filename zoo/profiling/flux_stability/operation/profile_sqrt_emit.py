"""Flux-stability profile for the sqrt_emit streaming operation."""

from napl.sim.operation import sqrt_emit

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for sqrt_emit, one per supported polarity."""
    # The radicand stays in [0, 1] so the square root is real and lands in [0, 1],
    # inside the legal range for both polarities.
    return [
        profile_op(
            sqrt_emit,
            ctor={'polarity': 'unipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 4},
            ],
            reference=lambda values, polarity: values[0].sqrt(),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            sqrt_emit,
            ctor={'polarity': 'bipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,), 'dim': 4},
            ],
            reference=lambda values, polarity: values[0].sqrt(),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

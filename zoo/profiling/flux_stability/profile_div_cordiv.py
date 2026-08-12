"""Flux-stability profile for the div_cordiv streaming operation."""

from napl.sim.operation import div_cordiv

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for div_cordiv, one per supported polarity."""
    # Both operands share Sobol dim 1 so the streams stay correlated, and the
    # dividend stays below the divisor floor so the quotient stays in [0, 1].
    return [
        profile_op(
            div_cordiv,
            ctor={'depth': 2, 'generator': 'Sobol', 'polarity': 'unipolar'},
            inputs=[
                {'range': (0.0, 0.25), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 1},
                {'range': (0.25, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 1},
            ],
            reference=lambda values, polarity: values[0] / values[1],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

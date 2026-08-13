"""Flux-stability profile for the mul_unibi streaming operation."""

from napl.sim.operation import mul_unibi

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for mul_unibi (unipolar times bipolar), one per supported polarity."""
    return [
        profile_op(
            mul_unibi,
            ctor={'polarity': 'bipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0] * values[1],
            output_polarity='bipolar',
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

"""Flux-stability profile for the mul_scale streaming operation."""

from napl.sim.operation import mul_scale

from profile_common import profile_op


SCALE = 2


def profile():
    """Return the flux-stability result dicts for mul_scale, one per supported polarity."""
    # A single encoded stream is multiplied by the fixed power-of-two scale of 2. The input is
    # narrowed to half range so x * 2 stays in the legal rate range for both polarities.
    return [
        profile_op(
            mul_scale,
            ctor={'polarity': 'unipolar', 'scale': SCALE, 'intwidth': 12, 'fracwidth': 4},
            inputs=[
                {'range': (0.0, 0.5), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 1},
            ],
            reference=lambda values, polarity: values[0] * SCALE,
            timesteps=256,
            seed=0,
        ),
        profile_op(
            mul_scale,
            ctor={'polarity': 'bipolar', 'scale': SCALE, 'intwidth': 12, 'fracwidth': 4},
            inputs=[
                {'range': (-0.5, 0.5), 'polarity': 'bipolar', 'shape': (16384,), 'dim': 1},
            ],
            reference=lambda values, polarity: values[0] * SCALE,
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

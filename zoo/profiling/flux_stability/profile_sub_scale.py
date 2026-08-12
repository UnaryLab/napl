"""Flux-stability profile for the sub_scale streaming operation."""

from napl.sim.operation import sub_scale

from profile_common import profile_op

SCALE = 2


def profile():
    """Return the flux-stability result dicts for sub_scale, one per supported polarity."""
    # Bipolar only: the negate-then-add mechanism has no unipolar form. Inputs
    # narrowed to [-0.9, 0.9] keep (a - b) / scale inside the bipolar range.
    return [
        profile_op(
            sub_scale,
            ctor={'polarity': 'bipolar', 'scale': SCALE, 'intwidth': 20, 'fracwidth': 4},
            inputs=[
                {'range': (-0.9, 0.9), 'polarity': 'bipolar', 'shape': (4096,)},
                {'range': (-0.9, 0.9), 'polarity': 'bipolar', 'shape': (4096,)},
            ],
            reference=lambda values, polarity: (values[0] - values[1]) / SCALE,
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

"""Flux-stability profile for the signabs_shiftreg streaming operation."""

from napl.sim.operation import signabs_shiftreg

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for signabs_shiftreg, one per supported polarity."""
    # Multi-output: the sign stream decodes to -sign(v) and the magnitude to |v|.
    return [
        profile_op(
            signabs_shiftreg,
            ctor={'depth': 8, 'polarity': 'bipolar'},
            inputs=[{'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: (-values[0].sign(), values[0].abs()),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

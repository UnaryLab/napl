"""Flux-stability profile for the pow_regen streaming operation."""

from napl.sim.operation import pow_regen

from profile_common import profile_op


def profile():
    """Return the unipolar flux-stability result for default cubic pow_regen."""
    return [
        profile_op(
            pow_regen,
            ctor={'polarity': 'unipolar', 'n': 3},
            inputs=[{
                'range': (0.0, 1.0),
                'polarity': 'unipolar',
                'shape': (16384,),
                'dim': 1,
            }],
            reference=lambda values, polarity: values[0].pow(3),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for result in profile():
        print(result)

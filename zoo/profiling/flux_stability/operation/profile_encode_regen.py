"""Flux-stability profile for the encode_regen streaming operation."""

from napl.sim.operation import encode_regen

from profile_common import profile_op


def profile():
    """Return the unipolar flux-stability result for default encode_regen."""
    return [
        profile_op(
            encode_regen,
            ctor={'polarity': 'unipolar'},
            inputs=[{
                'range': (0.0, 1.0),
                'polarity': 'unipolar',
                'shape': (16384,),
                'dim': 1,
            }],
            reference=lambda values, polarity: values[0],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for result in profile():
        print(result)

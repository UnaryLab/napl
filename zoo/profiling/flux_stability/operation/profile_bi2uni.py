"""Flux-stability profile for the bi2uni streaming operation."""

from napl.sim.operation import bi2uni

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for bi2uni, one per supported polarity."""
    # Converter: bipolar source stream, unipolar target stream, same value.
    return [
        profile_op(
            bi2uni,
            ctor={'width': 2, 'polarity': 'bipolar'},
            inputs=[{'range': (0.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: values[0],
            output_polarity='unipolar',
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

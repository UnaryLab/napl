"""Flux-stability profile for the uni2bi streaming operation."""

from napl.sim.operation import uni2bi

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for uni2bi, one per supported polarity."""
    # Converter: unipolar source stream, bipolar target stream, same value.
    return [
        profile_op(
            uni2bi,
            ctor={'width': 3, 'polarity': 'unipolar'},
            inputs=[{'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: values[0],
            output_polarity='bipolar',
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

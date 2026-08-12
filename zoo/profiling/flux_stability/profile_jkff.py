"""Flux-stability profile for the jkff streaming operation."""

from napl.sim.operation import jkff

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for jkff, one per supported polarity."""
    # Unipolar only: the j / (j + k) reference is a rate formula valid for non-negative rates,
    # and a bipolar j + k near zero makes it blow up.
    # Independent Bernoulli j, k streams drive the output to the steady rate j / (j + k).
    return [
        profile_op(
            jkff,
            ctor={'polarity': 'unipolar'},
            inputs=[
                {'range': (0.2, 0.8), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.2, 0.8), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0] / (values[0] + values[1]),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

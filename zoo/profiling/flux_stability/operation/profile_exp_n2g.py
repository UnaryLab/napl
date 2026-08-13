"""Flux-stability profile for the exp_n2g streaming operation."""

from napl.sim.operation import exp_n2g

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for exp_n2g, one per supported polarity."""
    return [
        profile_op(
            exp_n2g,
            ctor={'depth': 5, 'gain': 1, 'polarity': 'bipolar'},
            # The FSM takes non-negative x for exp(-2x); the output stream is unipolar.
            inputs=[{'range': (0.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)}],
            reference=lambda values, polarity: (-2 * values[0]).exp(),
            output_polarity='unipolar',
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

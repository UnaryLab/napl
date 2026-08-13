"""Flux-stability profile for the exp_n1 streaming operation."""

from napl.sim.operation import exp_n1

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for exp_n1, one per supported polarity."""
    return [
        profile_op(
            exp_n1,
            ctor={'polarity': 'unipolar', 'timestep': 256, 'generator': 'sobol', 'dim': 1},
            # Dim 5 decorrelates the input from the kernel's internal dims 1..4.
            inputs=[{'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 5}],
            reference=lambda values, polarity: (-values[0]).exp(),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

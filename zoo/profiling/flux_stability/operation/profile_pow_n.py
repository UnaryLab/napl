"""Flux-stability profile for the pow_n streaming operation."""

from napl.sim.operation import pow_n

from profile_common import profile_op

# Power profiled, matching the pow_n test.
N = 3


def profile():
    """Return the flux-stability result dicts for pow_n, one per supported polarity."""
    return [
        profile_op(
            pow_n,
            ctor={'polarity': 'unipolar', 'n': N},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0].pow(N),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            pow_n,
            ctor={'polarity': 'bipolar', 'n': N},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0].pow(N),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

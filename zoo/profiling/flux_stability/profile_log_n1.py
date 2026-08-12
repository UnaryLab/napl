"""Flux-stability profile for the log_n1 streaming operation."""

from napl.sim.operation import log_n1

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for log_n1, one per supported polarity."""
    return [
        profile_op(
            log_n1,
            ctor={'polarity': 'unipolar', 'timestep': 256, 'generator': 'sobol', 'dim': 1},
            # Dim 5 decorrelates the input from the kernel's internal dims 1..4.
            inputs=[{'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 5}],
            # log_n1 approximates the five-term Maclaurin truncation of log(1 + x).
            reference=lambda values, polarity: (
                values[0] - values[0]**2 / 2 + values[0]**3 / 3
                - values[0]**4 / 4 + values[0]**5 / 5
            ),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

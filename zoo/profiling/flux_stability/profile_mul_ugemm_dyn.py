"""Flux-stability profile for the mul_ugemm_dyn streaming operation."""

from napl.sim.operation import mul_ugemm_dyn

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for mul_ugemm_dyn, one per supported polarity."""
    return [
        profile_op(
            mul_ugemm_dyn,
            ctor={'polarity': 'unipolar', 'width': 4, 'generator': 'sobol', 'dim': 1},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0] * values[1],
            timesteps=256,
            seed=0,
        ),
        profile_op(
            mul_ugemm_dyn,
            ctor={'polarity': 'bipolar', 'width': 4, 'generator': 'sobol', 'dim': 1},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0] * values[1],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

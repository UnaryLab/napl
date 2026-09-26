"""Flux-stability profile for the exp_m1_delay streaming operation."""

from napl.sim.operation import exp_m1_delay

from profile_common import profile_op


def profile():
    """Return the unipolar flux-stability result for order-5 exp_m1_delay."""
    return [
        profile_op(
            exp_m1_delay,
            ctor={'polarity': 'unipolar', 'order': 5},
            inputs=[{
                'range': (0.0, 1.0),
                'polarity': 'unipolar',
                'shape': (16384,),
                'dim': 1,
            }],
            reference=lambda values, polarity: (values[0] - 1).exp(),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for result in profile():
        print(result)

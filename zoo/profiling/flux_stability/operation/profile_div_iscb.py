"""Flux-stability profile for the div_iscb streaming operation."""

from napl.sim.operation import div_iscb

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for div_iscb, one per supported polarity."""
    # In both polarities the dividend stays within the |divisor| floor so the quotient
    # stays in the legal range: unipolar [0, 1] and bipolar [-1, 1].
    return [
        profile_op(
            div_iscb,
            ctor={'polarity': 'unipolar'},
            inputs=[
                {'range': (0.0, 0.25), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 1},
                {'range': (0.25, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 2},
            ],
            reference=lambda values, polarity: values[0] / values[1],
            timesteps=256,
            seed=0,
        ),
        profile_op(
            div_iscb,
            ctor={'polarity': 'bipolar'},
            inputs=[
                {'range': (-0.25, 0.25), 'polarity': 'bipolar', 'shape': (16384,), 'dim': 1},
                {'range': (0.25, 1.0), 'polarity': 'bipolar', 'shape': (16384,), 'dim': 2},
            ],
            reference=lambda values, polarity: values[0] / values[1],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

"""Flux-stability profile for the encode_cond streaming operation."""

from napl.sim.operation import encode_cond

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for encode_cond (input stream times generated value), one per supported polarity."""
    return [
        profile_op(
            encode_cond,
            ctor={'polarity': 'unipolar', 'timestep': 256, 'generator': 'sobol'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'encode': False, 'range': (0.0, 1.0), 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0] * values[1],
            apply=lambda op, spikes, values: op(spikes[0], values[1]),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            encode_cond,
            ctor={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol'},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
                {'encode': False, 'range': (-1.0, 1.0), 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0] * values[1],
            apply=lambda op, spikes, values: op(spikes[0], values[1]),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

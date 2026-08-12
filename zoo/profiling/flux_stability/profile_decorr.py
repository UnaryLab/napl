"""Flux-stability profile for the decorr streaming operation."""

from napl.sim.operation import decorr

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for decorr, one per supported polarity."""
    # Two correlated inputs; output_0 preserves the first stream value.
    return [
        profile_op(
            decorr,
            ctor={'polarity': 'unipolar', 'depth': 4, 'timestep': 256,
                  'generator': 'sys', 'seed': 7},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0],
            apply=lambda op, spikes, values: op(*spikes)[0],
            timesteps=256,
            seed=0,
        ),
        profile_op(
            decorr,
            ctor={'polarity': 'bipolar', 'depth': 4, 'timestep': 256,
                  'generator': 'sys', 'seed': 7},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0],
            apply=lambda op, spikes, values: op(*spikes)[0],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

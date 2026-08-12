"""Flux-stability profile for the add_scale streaming operation."""

from napl.sim.operation import add_scale

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for add_scale, one per supported polarity."""
    return [
        profile_op(
            add_scale,
            ctor={'polarity': 'unipolar', 'scale': 8, 'intwidth': 20, 'fracwidth': 4},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (2048, 8), 'reduce_dim': -1},
            ],
            reference=lambda values, polarity: values[0].mean(dim=-1),
            apply=lambda op, spikes, values: op(spikes[0], dim=-1),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            add_scale,
            ctor={'polarity': 'bipolar', 'scale': 8, 'intwidth': 20, 'fracwidth': 4},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (2048, 8), 'reduce_dim': -1},
            ],
            reference=lambda values, polarity: values[0].mean(dim=-1),
            apply=lambda op, spikes, values: op(spikes[0], dim=-1),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

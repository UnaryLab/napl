"""Flux-stability profile for the add_gaines streaming operation."""

from napl.sim.operation import add_gaines

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for add_gaines (scaled MUX mode), one per supported polarity."""
    return [
        profile_op(
            add_gaines,
            ctor={'polarity': 'unipolar', 'scaled': True, 'entry': 8,
                  'generator': 'sobol', 'dim': 5},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (8, 2048), 'reduce_dim': 0},
            ],
            reference=lambda values, polarity: values[0].mean(dim=0),
            apply=lambda op, spikes, values: op(spikes[0], dim=0),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            add_gaines,
            ctor={'polarity': 'bipolar', 'scaled': True, 'entry': 8,
                  'generator': 'sobol', 'dim': 5},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (8, 2048), 'reduce_dim': 0},
            ],
            reference=lambda values, polarity: values[0].mean(dim=0),
            apply=lambda op, spikes, values: op(spikes[0], dim=0),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

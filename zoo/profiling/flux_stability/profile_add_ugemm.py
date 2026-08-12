"""Flux-stability profile for the add_ugemm streaming operation."""

from napl.sim.operation import add_ugemm

from profile_common import profile_op


def _reference(values, polarity):
    lo = -1.0 if polarity == 'bipolar' else 0.0
    return (values[0].sum(dim=-1) / 8).clamp(lo, 1.0)


def profile():
    """Return the flux-stability result dicts for add_ugemm (scaled mode), one per supported polarity."""
    return [
        profile_op(
            add_ugemm,
            ctor={'polarity': 'unipolar', 'scaled': True, 'width': 16},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (2048, 8), 'reduce_dim': -1},
            ],
            reference=_reference,
            apply=lambda op, spikes, values: op(spikes[0], dim=-1),
            timesteps=256,
            seed=0,
        ),
        profile_op(
            add_ugemm,
            ctor={'polarity': 'bipolar', 'scaled': True, 'width': 16},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (2048, 8), 'reduce_dim': -1},
            ],
            reference=_reference,
            apply=lambda op, spikes, values: op(spikes[0], dim=-1),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

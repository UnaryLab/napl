"""Flux-stability profile for the add_desync streaming operation."""

from napl.sim.operation import add_desync

from profile_common import profile_op


def _reference(values, polarity):
    if polarity == 'bipolar':
        return (values[0] + values[1] + 1.0).clamp(max=1.0)
    return (values[0] + values[1]).clamp(max=1.0)


def profile():
    """Return the flux-stability result dicts for add_desync, one per supported polarity."""
    return [
        profile_op(
            add_desync,
            ctor={'polarity': 'unipolar', 'depth': 1},
            inputs=[
                {'range': (0.0, 0.75), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.0, 0.5), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            reference=_reference,
            timesteps=256,
            seed=0,
        ),
        profile_op(
            add_desync,
            ctor={'polarity': 'bipolar', 'depth': 1},
            inputs=[
                {'range': (-1.0, 0.5), 'polarity': 'bipolar', 'shape': (16384,)},
                {'range': (-1.0, 0.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            reference=_reference,
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

"""Flux-stability profile for the subabs streaming operation."""

from napl.sim.operation import subabs

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for subabs on SCC +1 unipolar operands, one per supported polarity."""
    return [
        profile_op(
            subabs,
            ctor={'polarity': 'unipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 1},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 1},
            ],
            reference=lambda values, polarity: (values[0] - values[1]).abs(),
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

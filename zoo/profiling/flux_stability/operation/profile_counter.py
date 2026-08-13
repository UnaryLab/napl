"""Flux-stability profile for the counter streaming operation."""

from napl.sim.operation import counter

from profile_common import profile_op

# Running count saturates at this ceiling for width 3.
_WIDTH = 3
_MAX_COUNT = 2 ** _WIDTH - 1
_TIMESTEPS = 256


def profile():
    """Return the flux-stability result dicts for counter, one per supported polarity."""
    # A running spike count is non-negative, so only unipolar is supported.
    return [
        profile_op(
            counter,
            ctor={'width': _WIDTH, 'polarity': 'unipolar'},
            inputs=[{'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)}],
            # For rate p, the saturation flag decodes to 1 - max_count / (p * T).
            reference=lambda values, polarity: (
                1.0 - _MAX_COUNT / (values[0].clamp_min(1e-9) * _TIMESTEPS)
            ).clamp(0.0, 1.0),
            timesteps=_TIMESTEPS,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

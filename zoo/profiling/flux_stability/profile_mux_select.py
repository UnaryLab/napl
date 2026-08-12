"""Flux-stability profile for the mux_select streaming operation."""

from napl.sim.operation import mux_select

from profile_common import profile_op


# The select stream c is always unipolar (a mixing-weight rate); the operands a, b
# and the output follow the configured polarity. reference is the value-domain
# select p_c * a + (1 - p_c) * b, with c = values[0].
def profile():
    """Return the flux-stability result dicts for mux_select, one per supported polarity."""
    return [
        profile_op(
            mux_select,
            ctor={'polarity': 'unipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0] * values[1] + (1.0 - values[0]) * values[2],
            timesteps=256,
            seed=0,
        ),
        profile_op(
            mux_select,
            ctor={'polarity': 'bipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0] * values[1] + (1.0 - values[0]) * values[2],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

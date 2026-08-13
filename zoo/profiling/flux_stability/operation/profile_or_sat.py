"""Flux-stability profile for the or_sat streaming operation."""

from napl.sim.operation import or_sat

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for or_sat, one per supported polarity."""
    # Unipolar only: a bitwise OR of two bipolar streams has no saturating-add
    # value semantics. The two operands sit on distinct Sobol dims (default 1 and
    # 2) so a + b - a*b holds.
    return [
        profile_op(
            or_sat,
            ctor={'polarity': 'unipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0] + values[1] - values[0] * values[1],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

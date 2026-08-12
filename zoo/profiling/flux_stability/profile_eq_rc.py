"""Flux-stability profile for the eq_rc streaming operation."""

from napl.sim.operation import eq_rc

from profile_common import profile_op


# Rate-equality band, in decoded value units, matching the profiled operation.
_TOLERANCE = 0.1


def profile():
    """Return the flux-stability result dicts for eq_rc, one per supported polarity."""
    return [
        profile_op(
            eq_rc,
            ctor={'polarity': 'unipolar', 'tolerance': _TOLERANCE},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            # eq_rc emits a unipolar {0, 1} indicator for |value_0 - value_1| <= tolerance.
            reference=lambda values, polarity: ((values[0] - values[1]).abs() <= _TOLERANCE).float(),
            output_polarity='unipolar',
            timesteps=256,
            seed=0,
        ),
        profile_op(
            eq_rc,
            ctor={'polarity': 'bipolar', 'tolerance': _TOLERANCE},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            # eq_rc emits a unipolar {0, 1} indicator for |value_0 - value_1| <= tolerance.
            reference=lambda values, polarity: ((values[0] - values[1]).abs() <= _TOLERANCE).float(),
            output_polarity='unipolar',
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

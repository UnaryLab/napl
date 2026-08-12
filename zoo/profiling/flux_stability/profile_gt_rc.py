"""Flux-stability profile for the gt_rc streaming operation."""

from napl.sim.operation import gt_rc

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for gt_rc, one per supported polarity."""
    return [
        profile_op(
            gt_rc,
            ctor={'polarity': 'unipolar'},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            # gt_rc emits a unipolar {0, 1} indicator for input_0 > input_1.
            reference=lambda values, polarity: (values[0] > values[1]).float(),
            # The first timestep returns the scalar initial decision; broadcast it to the input shape.
            apply=lambda op, spikes, values: op(*spikes).expand_as(spikes[0]),
            output_polarity='unipolar',
            timesteps=256,
            seed=0,
        ),
        profile_op(
            gt_rc,
            ctor={'polarity': 'bipolar'},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            # gt_rc emits a unipolar {0, 1} indicator for input_0 > input_1.
            reference=lambda values, polarity: (values[0] > values[1]).float(),
            # The first timestep returns the scalar initial decision; broadcast it to the input shape.
            apply=lambda op, spikes, values: op(*spikes).expand_as(spikes[0]),
            output_polarity='unipolar',
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

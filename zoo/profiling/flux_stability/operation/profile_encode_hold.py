"""Flux-stability profile for the encode_hold streaming operation."""

from napl.sim.operation import encode_hold

from profile_common import profile_op


def profile():
    """Return the flux-stability result dicts for encode_hold, one per supported polarity."""
    # The trigger is placed at half the run so the profiled stream keeps both
    # phases: a pass-through first half and a re-emission of the frozen value in
    # the second. Both are stable streams of the same input value, so the flux is
    # a meaningful finite ratio rather than degenerate; it reflects a reused
    # stable value, not a computed transform.
    ctor_uni = {'polarity': 'unipolar', 'timestep': 256, 'trigger_timestep': 128}
    ctor_bi = {'polarity': 'bipolar', 'timestep': 256, 'trigger_timestep': 128}
    return [
        profile_op(
            encode_hold,
            ctor=ctor_uni,
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0],
            timesteps=256,
            seed=0,
        ),
        profile_op(
            encode_hold,
            ctor=ctor_bi,
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            reference=lambda values, polarity: values[0],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

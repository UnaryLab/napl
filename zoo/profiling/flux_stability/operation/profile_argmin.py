"""Flux-stability profile for the argmin streaming operation."""

import torch

from napl.sim.operation import argmin

from profile_common import profile_op


def _one_hot_argmin(values, polarity):
    """Return the numeric one-hot minimum along the candidate dimension."""
    winner = values[0].argmin(dim=-1, keepdim=True)
    return torch.zeros_like(values[0]).scatter_(-1, winner, 1)


def profile():
    """Return argmin flux-stability results for both input polarities."""
    return [
        profile_op(
            argmin,
            ctor={
                'polarity': polarity, 'dim': -1, 'width': 16,
            },
            inputs=[{
                'range': value_range,
                'polarity': polarity,
                'shape': (4096, 3),
                'reduce_dim': -1,
            }],
            reference=_one_hot_argmin,
            apply=lambda op, spikes, values: op(spikes[0]),
            output_polarity='unipolar',
            timesteps=256,
            seed=0,
        )
        for polarity, value_range in [
            ('unipolar', (0.0, 1.0)),
            ('bipolar', (-1.0, 1.0)),
        ]
    ]


if __name__ == '__main__':
    for result in profile():
        print(result)

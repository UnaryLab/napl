"""Flux-stability profile for the argmax streaming operation."""

import torch

from napl.sim.operation import argmax

from profile_common import profile_op


def _one_hot_argmax(values, polarity):
    """Return the numeric one-hot maximum along the candidate dimension."""
    winner = values[0].argmax(dim=-1, keepdim=True)
    return torch.zeros_like(values[0]).scatter_(-1, winner, 1)


def profile():
    """Return argmax flux-stability results for both input polarities."""
    return [
        profile_op(
            argmax,
            ctor={
                'polarity': polarity, 'dim': -1, 'width': 16,
            },
            inputs=[{
                'range': value_range,
                'polarity': polarity,
                'shape': (4096, 3),
                'reduce_dim': -1,
            }],
            reference=_one_hot_argmax,
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

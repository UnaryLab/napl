"""Flux-stability profile for the div_gaines streaming operation."""

import torch

from napl.sim.operation import div_gaines

from profile_common import profile_op


def _signed_divisor(shape):
    """Draw a bipolar divisor with random sign and magnitude in [0.25, 1.0) so the quotient stays legal."""
    magnitude = 0.25 + 0.75 * torch.rand(shape)
    sign = torch.where(torch.rand(shape) < 0.5, -1.0, 1.0)
    return sign * magnitude


def profile():
    """Return the flux-stability result dicts for div_gaines, one per supported polarity."""
    # In both polarities the dividend stays within the |divisor| floor so the quotient stays
    # in the legal range: unipolar [0, 1] and bipolar [-1, 1]; the bipolar divisor spans both signs.
    return [
        profile_op(
            div_gaines,
            ctor={'polarity': 'unipolar', 'width': 5, 'generator': 'sobol', 'dim': 3},
            inputs=[
                {'range': (0.0, 0.25), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 1},
                {'range': (0.25, 1.0), 'polarity': 'unipolar', 'shape': (16384,), 'dim': 2},
            ],
            reference=lambda values, polarity: values[0] / values[1],
            timesteps=256,
            seed=0,
        ),
        profile_op(
            div_gaines,
            ctor={'polarity': 'bipolar', 'width': 5, 'generator': 'sobol', 'dim': 3},
            inputs=[
                {'range': (-0.25, 0.25), 'polarity': 'bipolar', 'shape': (16384,), 'dim': 1},
                {'polarity': 'bipolar', 'shape': (16384,), 'dim': 2, 'sample': _signed_divisor},
            ],
            reference=lambda values, polarity: values[0] / values[1],
            timesteps=256,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

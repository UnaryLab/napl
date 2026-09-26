"""Flux-stability profile for the eq streaming operation."""

import torch

from napl.sim.operation import eq

from profile_common import profile_op


# Counter bit width and equality band, in counter steps, matching the profiled operation.
_WIDTH = 3
_TOLERANCE = 1
_TIMESTEPS = 256


def _reference(values, polarity):
    """Fraction of the run the counter stays inside its band, per element."""
    # The counter drifts by the rate gap per timestep, so it leaves the band
    # after tolerance / gap timesteps and a persistent gap then holds it on a
    # rail: the decoded unipolar output is min(1, tolerance / (gap * timesteps)).
    gap = (values[0] - values[1]).abs()
    if polarity == 'bipolar':
        # A bipolar value v rides a rate (v + 1) / 2, halving the rate gap.
        gap = gap / 2
    band = torch.full_like(gap, float(_TOLERANCE))
    inside = torch.where(gap > 0, band / (gap * _TIMESTEPS), torch.ones_like(gap))
    return inside.clamp(max=1.0).float()


def profile():
    """Return the flux-stability result dicts for eq, one per supported polarity."""
    return [
        profile_op(
            eq,
            ctor={'polarity': 'unipolar', 'width': _WIDTH, 'tolerance': _TOLERANCE},
            inputs=[
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
                {'range': (0.0, 1.0), 'polarity': 'unipolar', 'shape': (16384,)},
            ],
            reference=_reference,
            output_polarity='unipolar',
            timesteps=_TIMESTEPS,
            seed=0,
        ),
        profile_op(
            eq,
            ctor={'polarity': 'bipolar', 'width': _WIDTH, 'tolerance': _TOLERANCE},
            inputs=[
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
                {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (16384,)},
            ],
            reference=_reference,
            output_polarity='unipolar',
            timesteps=_TIMESTEPS,
            seed=0,
        ),
    ]


if __name__ == '__main__':
    for r in profile():
        print(r)

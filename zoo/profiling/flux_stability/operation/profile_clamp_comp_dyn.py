"""Flux-stability profile for the clamp_comp_dyn streaming operation."""

import torch

from napl.sim.operation import clamp_comp_dyn

from profile_common import profile_op

# Bands per polarity: the dyadic band profile_clamp_dyn.py uses, so the two entries
# read against each other, and a second band whose lower bound sits off the 2**-8
# probability grid a 256-timestep bound stream is encoded on. Each profiled level is
# the mean over both bands, so the entry is not sampled at one grid-aligned point.
OFF_GRID = 3.0 / 1024.0
BANDS = {
    'bipolar': ((-0.5, 0.5), (-0.5 + OFF_GRID, 0.5)),
    'unipolar': ((0.25, 0.75), (0.25 + OFF_GRID, 0.75)),
}

# Metrics averaged over the band set; every other result field is shared by the runs.
_MEAN_KEYS = ('flux_stability', 'input_stability', 'rmse')


def mean_over_bands(runs):
    """Return one result dict holding each profiled metric's mean over the band set."""
    merged = dict(runs[0])
    for key in _MEAN_KEYS:
        merged[key] = sum(run[key] for run in runs) / len(runs)
    return merged


def _constant(value):
    """Return a sampler placing one constant bound value in the spec's shape."""
    return lambda shape: torch.full(shape, value)


def profile():
    """Return the flux-stability result dicts for clamp_comp_dyn, one per supported polarity."""
    results = []
    for polarity in ('bipolar', 'unipolar'):
        rng = (-1.0, 1.0) if polarity == 'bipolar' else (0.0, 1.0)
        runs = [
            profile_op(
                clamp_comp_dyn,
                ctor={'polarity': polarity},
                inputs=[
                    {'range': rng, 'polarity': polarity, 'shape': (16384,), 'dim': 1},
                    # The two bound streams ride Sobol dimensions distinct from each
                    # other and from the input, per the class's documented contract.
                    {'range': (lo, lo), 'sample': _constant(lo), 'polarity': polarity,
                     'shape': (1,), 'dim': 2},
                    {'range': (hi, hi), 'sample': _constant(hi), 'polarity': polarity,
                     'shape': (1,), 'dim': 3},
                ],
                reference=lambda values, _polarity: torch.clamp(values[0], values[1], values[2]),
                timesteps=256,
                seed=0,
            )
            for lo, hi in BANDS[polarity]
        ]
        results.append(mean_over_bands(runs))
    return results


if __name__ == '__main__':
    for r in profile():
        print(r)

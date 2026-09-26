"""Flux-stability profile for the clamp_sat streaming operation."""

import torch

from napl.sim.operation import clamp_sat

from profile_common import profile_op

# Bands per polarity: the dyadic band profile_clamp_comp.py uses, so the two entries
# read against each other, and a second band whose lower bound sits off the 2**-8
# value grid the band constants are quantized to. Each profiled level is the mean
# over both bands, so the entry is not sampled at one grid-aligned point.
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


def profile():
    """Return the flux-stability result dicts for clamp_sat, one per supported polarity."""
    results = []
    for polarity in ('bipolar', 'unipolar'):
        rng = (-1.0, 1.0) if polarity == 'bipolar' else (0.0, 1.0)
        runs = [
            profile_op(
                clamp_sat,
                ctor={'polarity': polarity, 'lo': lo, 'hi': hi},
                inputs=[{'range': rng, 'polarity': polarity, 'shape': (16384,)}],
                reference=lambda values, _polarity, lo=lo, hi=hi: torch.clamp(values[0], lo, hi),
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

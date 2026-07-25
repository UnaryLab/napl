import torch

from napl.sim.base import napl_base
from napl.sim.metric._shared import analyze
from napl.sim.metric.stability import stability
from loguru import logger


class stability_flux(napl_base):
    """
    Element-wise ratio of the stabilities of two spike streams:
    flux = stability(stream 1) / stability(stream 2). flux > 1 means stream 1 is
    more stable than stream 2. A denominator stability of 0 yields inf (raw torch
    division semantics, deliberately not clamped). Call forward(spike_1, spike_2)
    once per timestep, then analyze() for the final ratio.
    """
    def __init__(
            self,
            source_1,
            source_2,
            config={
                'polarity': 'bipolar',
                'threshold': 0.05,
            }
        ):
        super().__init__(config, ['polarity', 'threshold'], polarity_required=True)

        self.stability_1 = stability(source_1, config)
        self.stability_2 = stability(source_2, config)

    def forward(self, spike_1, spike_2):
        self.stability_1(spike_1)
        self.stability_2(spike_2)
        # no return: readers access .flux on demand.


    @property
    def flux(self):
        """
        Stability ratio, computed on access from the two inner monitors.
        Returns a fresh tensor; before any forward() it is the zeros seed.
        """
        if not self.valid:
            return torch.zeros_like(self.stability_1.source)
        return self.stability_1.stability / self.stability_2.stability


    def analyze(self, verbose=False):
        # return the flux and index of max abs flux
        assert self.valid, logger.error('Metric is not valid. Please call forward() before analyze().')
        # one property access: flux computes from the two inner monitors on each read
        flux = self.flux
        result = analyze(
            flux,
            verbose=verbose,
            report='Flux Stability',
            value='flux stability',
            timestep=self.timestep_cur,
        )
        self.flux_abs_max = result.absolute_max
        self.flux_abs_min = result.absolute_min
        self.flux_avg = result.mean
        self.flux_mae = result.mean_absolute
        self.flux_rmse = result.root_mean_square

        return flux, result.max_absolute_index

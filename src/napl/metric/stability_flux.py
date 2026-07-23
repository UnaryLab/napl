import torch

from napl.base import napl_base
from napl.metric.stability import stability
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


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.stability_1.reset()
        self.stability_2.reset()


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
        assert self.valid, logger.error(f'Metric is not valid. Please call forward() before analyze().')
        # one property access: flux computes from the two inner monitors on each read
        flux = self.flux
        # abs() is reused below; compute once to avoid redundant full-tensor passes.
        flux_abs = flux.abs()
        # aminmax: one fused pass for both extremes instead of separate min/max scans.
        flux_abs_amin, flux_abs_amax = torch.aminmax(flux_abs)
        self.flux_abs_max = flux_abs_amax
        self.flux_abs_min = flux_abs_amin
        self.flux_avg = flux.mean()
        self.flux_mae = flux_abs.mean()
        self.flux_rmse = torch.sqrt(flux_abs.pow(2).mean())

        if verbose:
            logger.info(f'Stability flux report for stability_flux instance <{self.name}> over <{self.timestep_cur}> timesteps: ')
            logger.info(f'    Max absolute flux:     <{self.flux_abs_max.item()}>')
            logger.info(f'    Min absolute flux:     <{self.flux_abs_min.item()}>')
            logger.info(f'    Mean flux:             <{self.flux_avg.item()}>')
            logger.info(f'    Mean absolute flux:    <{self.flux_mae.item()}>')
            logger.info(f'    Root mean square flux: <{self.flux_rmse.item()}>')
            logger.info(f'')
        return flux, torch.argmax(flux_abs)

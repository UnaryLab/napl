import torch

from napl.base import napl_base
from napl.metric.accuracy import accuracy
from loguru import logger


class stability(napl_base):
    """
    Per-element stability of a spike stream: the fraction of the run that remains after
    the progressive error last exceeded `threshold`. A stream whose error settles early
    scores near 1; one that stays noisy to the end scores near 0. Call forward(spike)
    once per timestep, then analyze() for the final stability. References: uGEMM;
    "Normalized Stability: A Cross-Level Design Metric for Early Termination in
    Stochastic Computing".
    """
    def __init__(
            self,
            source,
            config={
                'polarity': 'bipolar',
                'threshold': 0.05,
            }
        ):
        super().__init__(config, ['polarity', 'threshold'], polarity_required=True)

        self.source = source
        self.threshold = config['threshold']
        # inner progressive-error monitor
        self.accuracy = accuracy({'polarity': self.polarity})
        # last timestep (per element) at which the error was still above threshold
        self.cycle_to_stable = torch.nn.Parameter(torch.zeros_like(source), requires_grad=False)


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.accuracy.reset()
        self.cycle_to_stable.data = torch.zeros_like(self.source)


    @property
    def stability(self):
        """
        Stability, computed on access from cycle_to_stable.
        Returns a fresh tensor; before any forward() it is the zeros seed.
        """
        if not self.valid:
            return torch.zeros_like(self.source)
        cycle = self.accuracy.timestep_cur
        return 1 - self.cycle_to_stable.clamp(1, cycle).div(cycle)


    def forward(self, spike):
        # accumulate progressive precision; only the per-element error is needed here, so
        # take it directly rather than via analyze (which also runs unused reductions).
        self.accuracy(spike)
        spike_value = self.accuracy.spike_value
        # per-element error vs source; sub_/abs_ in place on the fresh spike_value
        # (the spike_value property returns a new tensor on each access, so mutating
        # it here leaves self.source and accuracy's state untouched). Both
        # operands are float, so in-place subtraction does not change dtype.
        # mark this cycle as the last unstable one wherever the error exceeds threshold;
        # masked_fill_ is the in-place conditional assignment cycle_to_stable[mask] = cycle
        unstable = spike_value.sub_(self.source).abs_() > self.threshold
        self.cycle_to_stable.data.masked_fill_(unstable, self.accuracy.timestep_cur)
        # no return: readers access .stability on demand.


    def analyze(self, verbose=False):
        # return the stability and index of max abs stability
        assert self.valid, logger.error(f'Metric is not valid. Please call forward() before analyze().')
        # one property access: stability computes from cycle_to_stable on each read
        stability = self.stability
        # abs() is reused below; compute once to avoid redundant full-tensor passes.
        stability_abs = stability.abs()
        # aminmax: one fused pass for both extremes instead of separate min/max scans.
        stability_abs_amin, stability_abs_amax = torch.aminmax(stability_abs)
        self.stability_abs_max = stability_abs_amax
        self.stability_abs_min = stability_abs_amin
        self.stability_avg = stability.mean()
        self.stability_mae = stability_abs.mean()
        self.stability_rmse = torch.sqrt(stability_abs.pow(2).mean())

        if verbose:
            logger.info(f'Stability report for stability instance <{self.name}> over <{self.timestep_cur}> timesteps: ')
            logger.info(f'    Max absolute stability:     <{self.stability_abs_max.item()}>')
            logger.info(f'    Min absolute stability:     <{self.stability_abs_min.item()}>')
            logger.info(f'    Mean stability:             <{self.stability_avg.item()}>')
            logger.info(f'    Mean absolute stability:    <{self.stability_mae.item()}>')
            logger.info(f'    Root mean square stability: <{self.stability_rmse.item()}>')
            logger.info(f'')
        return stability, torch.argmax(stability_abs)

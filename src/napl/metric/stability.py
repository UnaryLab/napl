import torch

from napl.base import napl_base
from napl.metric._shared import analyze
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

        self.register_buffer('source', source)
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
        result = analyze(
            stability,
            verbose=verbose,
            report='Stability',
            value='stability',
            timestep=self.timestep_cur,
        )
        self.stability_abs_max = result.absolute_max
        self.stability_abs_min = result.absolute_min
        self.stability_avg = result.mean
        self.stability_mae = result.mean_absolute
        self.stability_rmse = result.root_mean_square

        return stability, result.max_absolute_index

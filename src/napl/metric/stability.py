import torch

from napl.base import napl_base
from napl.metric.accuracy import accuracy


class stability(napl_base):
    """
    Per-element stability of a spike stream: the fraction of the run that remains after
    the progressive error last exceeded `threshold`. A stream whose error settles early
    scores near 1; one that stays noisy to the end scores near 0. Call forward(spike)
    once per timestep, then report() for the final stability. References: uGEMM;
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
        self.stability = torch.nn.Parameter(torch.zeros_like(source), requires_grad=False)


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.accuracy.reset()
        self.cycle_to_stable.data = torch.zeros_like(self.source)
        self.stability.data = torch.zeros_like(self.source)


    def forward(self, spike):
        self.tick()
        # accumulate progressive precision; only the per-element error is needed here, so
        # take it directly rather than via report_error (which also runs unused reductions).
        spike_value = self.accuracy(spike)
        # per-element error vs source; sub_/abs_ in place on the fresh spike_value
        # (accuracy returns a new spike_value each call and overwrites it the next call,
        # so mutating it here leaves self.source and accuracy's state untouched). Both
        # operands are float, so in-place subtraction does not change dtype.
        # mark this cycle as the last unstable one wherever the error exceeds threshold;
        # masked_fill_ is the in-place conditional assignment cycle_to_stable[mask] = cycle
        unstable = spike_value.sub_(self.source).abs_() > self.threshold
        self.cycle_to_stable.data.masked_fill_(unstable, self.accuracy.timestep_cur)
        return self.stability


    def report_stab(self):
        cycle = self.accuracy.timestep_cur
        self.stability.data = 1 - self.cycle_to_stable.clamp(1, cycle).div(cycle)
        return self.stability

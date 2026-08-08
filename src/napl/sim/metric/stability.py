import torch

from napl.sim.base import napl_base
from napl.sim.metric._shared import analyze
from .accuracy import accuracy
from loguru import logger


class stability(napl_base):
    r"""
    Measure when each element of a spike stream settles near its source value.

    Stability is the fraction of the run remaining after progressive error last
    exceeded the configured threshold. Use it to compare early convergence: a
    stream that settles early approaches ``1``, while one that remains unstable
    approaches ``0``.

    Let :math:`\hat x_t` be the progressive decoded value, :math:`x` the source,
    and :math:`\theta` the threshold. The target is

    .. math::

       S_T = 1 - \frac{k_T}{T},\qquad
       k_T = \max\left(\{0\}\cup\{t \leq T : |\hat x_t - x| > \theta\}\right).

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import stability

        metric = stability(torch.tensor([1.0]))
        for _ in range(4):
            metric(torch.tensor([1.0]))
        value, result = metric.analyze()

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.

        *Normalized Stability: A Cross-Level Design Metric for Early Termination in Stochastic Computing*, ASP-DAC, 2021.
    """


    def __init__(
            self,
            source,
            config={
                'polarity': 'bipolar',
                'threshold': 0.05,
            }
        ):
        """
        Configure the reference value and stability threshold.

        If ``config`` is supplied, it must contain both ``polarity`` and
        ``threshold``.

        .. container:: api-parameter-list

            **Parameters:**

            - **source** – Tensor of expected decoded values. Use values in
              ``[0, 1]`` for unipolar streams or ``[-1, 1]`` for bipolar streams.
            - **config** – Configuration mapping. The default is
              ``{'polarity': 'bipolar', 'threshold': 0.05}``.

              - **polarity**: Stream encoding, either ``"unipolar"`` or
                ``"bipolar"``; the default is ``"bipolar"``.
              - **threshold**: Maximum absolute progressive error considered
                stable; the default is ``0.05``.
              - **name**: Optional instance label; the default is ``None``.
        """
        super().__init__(config, ['polarity', 'threshold'], polarity_required=True)

        #: Expected decoded value used as the per-element stability reference.
        self.source: torch.Tensor
        self.register_buffer('source', source)
        #: Maximum absolute progressive error treated as stable.
        self.threshold = config['threshold']
        #: Progressive decoder and error monitor for the observed spike stream.
        self.accuracy = accuracy({'polarity': self.polarity})
        #: Last timestep at which each element exceeded :attr:`threshold`.
        self.cycle_to_stable: torch.Tensor
        self.register_buffer('cycle_to_stable', torch.zeros_like(source))


    def _reset(self):
        """
        Clear the class-local last-unstable-timestep values.

        The inherited reset method also resets the timestep and the child
        accuracy metric before calling this hook. This hook returns ``None``.
        """
        self.cycle_to_stable.zero_()


    def forward(self, spike):
        """
        Record one timestep and update the last unstable timestep per element.

        Args:
            spike: Current 0/1 spike tensor with the same logical shape as
                ``source``.

        Calling the metric increments its timestep, advances the child accuracy
        metric, and updates elements whose absolute progressive error is greater
        than ``threshold``. The method returns ``None``.

        **Example:**

        .. code-block:: python

            metric(torch.tensor([1.0]))
        """
        self.accuracy(spike)
        spike_value = self.accuracy.spike_value
        # spike_value is fresh float state, so in-place error math cannot alias stored data.
        unstable = spike_value.sub_(self.source).abs_() > self.threshold
        self.cycle_to_stable.masked_fill_(unstable, self.accuracy.timestep_cur)


    @property
    def stability(self):
        """
        Return the current per-element stability.

        The result is a fresh tensor with the source shape and values in
        ``[0, 1]``. Before the first timestep, it is all zeros. An element that
        never exceeds the threshold reports ``1 - 1/T`` after ``T`` timesteps
        rather than ``1``. Reading this property does not change metric state.

        **Example:**

        .. code-block:: python

            current = metric.stability
        """
        if not self.valid:
            return torch.zeros_like(self.source)
        cycle = self.accuracy.timestep_cur
        return 1 - self.cycle_to_stable.clamp(1, cycle).div(cycle)


    def analyze(self, verbose=False):
        """
        Summarize the current per-element stability values.

        Call this method after at least one timestep.

        Args:
            verbose: Set to ``True`` to print the analysis summary. The default
                is ``False``.

        Returns:
            A pair containing the per-element stability tensor and its complete
            :class:`napl.sim.metric._shared.Analysis` summary.

        This method does not change the accumulated metric state.

        **Example:**

        .. code-block:: python

            value, result = metric.analyze()
        """
        if not self.valid:
            message = 'Metric is not valid. Please call forward() before analyze().'
            logger.error(message)
            raise AssertionError(message)
        stability = self.stability
        result = analyze(
            stability,
            verbose=verbose,
            report='Stability',
            value='stability',
            timestep=self.timestep_cur,
        )
        return stability, result

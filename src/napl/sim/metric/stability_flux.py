import torch

from napl.sim.base import napl_base
from napl.sim.metric._shared import analyze
from .stability import stability
from loguru import logger


class stability_flux(napl_base):
    r"""
    Compare two spike streams through their element-wise stability ratio.

    Use this metric when relative stability matters more than either absolute
    stability. A ratio greater than ``1`` means the first stream is more stable.
    Division follows PyTorch semantics, so a zero denominator can produce
    ``inf`` or ``nan``.

    With :math:`S^{(1)}_T` and :math:`S^{(2)}_T` the stabilities of the two
    streams, the target is their ratio

    .. math::

       F_T = \frac{S^{(1)}_T}{S^{(2)}_T}.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.metric import stability_flux

        metric = stability_flux(torch.ones(1), torch.ones(1))
        for _ in range(2):
            metric(torch.ones(1), torch.ones(1))
        ratio, result = metric.analyze()

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
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
        """
        Configure the two source values and their shared stability definition.

        If ``config`` is supplied, it must contain both ``polarity`` and
        ``threshold``.

        .. container:: api-parameter-list

            **Parameters:**

            - **source_1** – Expected decoded-value tensor for the numerator
              stream.
            - **source_2** – Expected decoded-value tensor for the denominator
              stream.
            - **config** – Shared configuration for both stability monitors. The
              default is ``{'polarity': 'bipolar', 'threshold': 0.05}``.

              - **polarity**: Stream encoding, either ``"unipolar"`` or
                ``"bipolar"``; the default is ``"bipolar"``.
              - **threshold**: Maximum absolute progressive error considered
                stable; the default is ``0.05``.
              - **name**: Optional instance label; the default is ``None``.
        """
        super().__init__(config, ['polarity', 'threshold'], polarity_required=True)

        #: Stability monitor for the numerator spike stream.
        self.stability_1 = stability(source_1, config)
        #: Stability monitor for the denominator spike stream.
        self.stability_2 = stability(source_2, config)

        self.polarity_io = {'input_1': self.polarity, 'input_2': self.polarity}


    def _reset(self):
        """
        Perform the class-local reset, which has no additional mutable state.

        The inherited reset method resets the timestep and both child stability
        monitors before calling this hook. This hook returns ``None``.
        """
        pass


    def forward(self, input_1, input_2):
        """
        Record one timestep from each spike stream.

        Args:
            input_1: Current 0/1 spike tensor for the numerator stream.
            input_2: Current 0/1 spike tensor for the denominator stream.

        Calling the metric increments its timestep and advances both child
        stability monitors. The method returns ``None``.

        **Example:**

        .. code-block:: python

            metric(torch.ones(1), torch.ones(1))
        """
        self.stability_1(input_1)
        self.stability_2(input_2)


    @property
    def stability_flux(self):
        """
        Return the first stream's stability divided by the second stream's.

        The result is a fresh tensor. Before the first timestep, it is all zeros.
        Afterward, zero-denominator cases retain PyTorch's ``inf`` and ``nan``
        results. Reading this property does not change metric state.

        **Example:**

        .. code-block:: python

            current_ratio = metric.stability_flux
        """
        if not self.valid:
            return torch.zeros_like(self.stability_1.source)
        return self.stability_1.stability / self.stability_2.stability


    def analyze(self, verbose=False):
        """
        Summarize the current per-element stability ratios.

        Call this method after at least one timestep.

        Args:
            verbose: Set to ``True`` to print the analysis summary. The default
                is ``False``.

        Returns:
            A pair containing the per-element stability-ratio tensor and its
            complete ``Analysis`` summary.

        This method does not change the accumulated metric state.

        **Example:**

        .. code-block:: python

            ratio, result = metric.analyze()
        """
        if not self.valid:
            message = 'Metric is not valid. Please call forward() before analyze().'
            logger.error(message)
            raise AssertionError(message)
        stability_flux_val = self.stability_flux
        result = analyze(
            stability_flux_val,
            verbose=verbose,
            report='Flux Stability',
            value='flux stability',
            timestep=self.timestep_cur,
        )
        return stability_flux_val, result

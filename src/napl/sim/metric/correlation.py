import torch

from napl.sim.base import napl_base
from napl.sim.metric._shared import analyze
from loguru import logger


class correlation(napl_base):
    r"""
    Measure stochastic cross-correlation (SCC) between two spike streams.

    Use this metric to quantify correlation from bit-pair counts accumulated over
    time. Supplying only the first stream measures its one-timestep-delayed
    autocorrelation.

    Let ``a``, ``b``, ``c``, and ``d`` count the ``11``, ``10``, ``01``, and
    ``00`` bit pairs over ``n`` timesteps. The target is the SCC

    .. math::

       \mathrm{SCC} = \begin{cases}
       \dfrac{ad-bc}{n\min(a+b,\,a+c)-(a+b)(a+c)}, & ad > bc,\\[1.2ex]
       \dfrac{ad-bc}{(a+b)(a+c)-n\max(a-d,\,0)}, & ad \leq bc.
       \end{cases}

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import correlation

        metric = correlation()
        stream = torch.tensor([1.0, 1.0, 0.0, 0.0])
        for spike in stream:
            metric(spike, spike)
        value, result = metric.analyze()

    .. container:: api-references

        .. rubric:: References

        *Exploiting correlation in stochastic circuit design*, ICCD, 2013.
    """


    def __init__(
            self,
            config={}
        ):
        """
        Construct an empty SCC accumulator.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Optional base configuration mapping; the default is
              ``{}``.

              - **name**: Optional instance label; the default is ``None``.
        """
        super().__init__(config, [])

        # These three counts determine the full 11/10/01/00 pair histogram.
        #: Running count of timesteps where both input streams contain one.
        self.paired_11: torch.Tensor
        self.register_buffer('paired_11', torch.zeros(1, dtype=self.ntype))
        #: Running count of one-valued spikes in the first input stream.
        self.sum_1: torch.Tensor
        self.register_buffer('sum_1', torch.zeros(1, dtype=self.ntype))
        #: Running count of one-valued spikes in the second input stream.
        self.sum_2: torch.Tensor
        self.register_buffer('sum_2', torch.zeros(1, dtype=self.ntype))
        #: Previous first-input spike tensor used in single-input autocorrelation mode.
        self.input_1_d: torch.Tensor
        self.register_buffer('input_1_d', torch.zeros(1, dtype=self.ntype))


    def _reset(self):
        """
        Clear all local bit-pair counts and the autocorrelation delay value.

        Each buffer returns to a scalar zero seed. The inherited reset method
        resets the timestep before calling this hook. This hook returns ``None``.
        """
        for p in [self.paired_11, self.sum_1, self.sum_2, self.input_1_d]:
            p.resize_(1).zero_()


    def forward(self, input_1, input_2=None):
        """
        Accumulate one pair of spike-stream timesteps.

        Args:
            input_1: First 0/1 spike tensor for the current timestep.
            input_2: Second 0/1 spike tensor with a broadcast-compatible shape.
                When ``None``, use the one-timestep-delayed first input. The
                default is ``None``.

        Calling the metric increments its timestep and updates the joint counts.
        In single-input mode it also stores ``input_1`` for the next call. The
        method returns ``None``.

        **Example:**

        .. code-block:: python

            metric(torch.tensor([1.0]), torch.tensor([0.0]))
        """
        if input_2 is None:
            input_2 = self.input_1_d.clone().detach()
            input_1_d = input_1.detach().type(self.ntype)
            if self.input_1_d.shape == input_1_d.shape:
                self.input_1_d.copy_(input_1_d)
            else:
                self.input_1_d.resize_as_(input_1_d).copy_(input_1_d)

        # addcmul and add promote bool inputs to the accumulator dtype exactly.
        input_1_is_1 = torch.ne(input_1, 0)
        input_2_is_1 = torch.ne(input_2, 0)

        if self.paired_11.shape == input_1_is_1.shape:
            self.paired_11.addcmul_(input_1_is_1, input_2_is_1)
            self.sum_1.add_(input_1_is_1)
            self.sum_2.add_(input_2_is_1)
        else:
            # Scalar seeds broadcast to the input shape on the first call.
            paired_11 = torch.addcmul(self.paired_11, input_1_is_1, input_2_is_1).detach()
            sum_1 = self.sum_1.add(input_1_is_1).detach()
            sum_2 = self.sum_2.add(input_2_is_1).detach()
            self.paired_11.resize_as_(paired_11).copy_(paired_11)
            self.sum_1.resize_as_(sum_1).copy_(sum_1)
            self.sum_2.resize_as_(sum_2).copy_(sum_2)


    @property
    def correlation(self):
        """
        Return the SCC computed from the accumulated bit-pair counts.

        Values range from ``-1`` for fully anticorrelated streams to ``1`` for
        fully correlated streams. The denominator is floored at ``1``, so empty
        or degenerate counts stay finite. Reading this property does not change
        metric state.

        **Example:**

        .. code-block:: python

            coefficient = metric.correlation
        """
        # Pair counts are a=11, b=10, c=01, and d=00; n is the run length.
        a = self.paired_11
        b = self.sum_1 - a
        c = self.sum_2 - a
        n = self.timestep_cur
        d = n - a - b - c
        ad_minus_bc = a * d - b * c
        ad_gt_bc = torch.gt(ad_minus_bc, 0).type(self.ntype)
        ad_le_bc = 1 - ad_gt_bc
        a_plus_b = a + b
        a_plus_c = a + c
        a_minus_d = a - d
        zeros = torch.zeros_like(a)
        ones = torch.ones_like(a)
        # The SCC denominator depends on the sign of ad-bc and is clamped away from zero.
        corr_gt = ad_minus_bc.div(torch.max(torch.min(a_plus_b, a_plus_c) * n - a_plus_b * a_plus_c, ones))
        corr_le = ad_minus_bc.div(torch.max(a_plus_b * a_plus_c - torch.max(a_minus_d, zeros) * n, ones))
        return ad_gt_bc * corr_gt + ad_le_bc * corr_le


    def analyze(self, verbose=False):
        """
        Summarize the current per-element SCC values.

        Call this method after at least one timestep.

        Args:
            verbose: Set to ``True`` to print the analysis summary. The default
                is ``False``.

        Returns:
            A pair containing the per-element SCC tensor and its complete
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
        correlation = self.correlation
        result = analyze(
            correlation,
            verbose=verbose,
            report='Correlation',
            value='correlation',
            timestep=self.timestep_cur,
        )
        return correlation, result

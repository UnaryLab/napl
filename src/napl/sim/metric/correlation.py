import torch

from napl.sim.base import napl_base
from napl.sim.metric._shared import analyze
from loguru import logger


class correlation(napl_base):
    """
    Measure stochastic cross-correlation (SCC) between two spike streams.

    Use this metric to quantify correlation from bit-pair counts accumulated over
    time. Supplying only the first stream measures its one-timestep-delayed
    autocorrelation.

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

        *Exploiting Correlation in Stochastic Circuit Design*.
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

        # sufficient statistics for the joint bit-pair histogram (a=11, b=10, c=01, d=00):
        # the co-occurrence count and the per-stream 1-counts. b, c, d and the run length
        # are all recovered at report() from these plus timestep_cur, so each step accumulates
        # only three counts and forms one product.
        self.register_buffer('paired_11', torch.zeros(1, dtype=self.ntype))
        self.register_buffer('sum_1', torch.zeros(1, dtype=self.ntype))
        self.register_buffer('sum_2', torch.zeros(1, dtype=self.ntype))
        # one-step delay buffer for the autocorrelation (single-input) case
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

        # bool is left uncast: addcmul/add promote it to the ntype accumulator, so the
        # two per-timestep .type() casts are redundant dispatches (kept as int8/float 0/1).
        input_1_is_1 = torch.ne(input_1, 0)
        input_2_is_1 = torch.ne(input_2, 0)

        if self.paired_11.shape == input_1_is_1.shape:
            self.paired_11.addcmul_(input_1_is_1, input_2_is_1)
            self.sum_1.add_(input_1_is_1)
            self.sum_2.add_(input_2_is_1)
        else:
            # The scalar seeds broadcast to the input shape on the first call.
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
        fully correlated streams. Empty or degenerate counts use the guarded
        denominators in the SCC definition. Reading this property does not
        change metric state.

        **Example:**

        .. code-block:: python

            coefficient = metric.correlation
        """
        a = self.paired_11               # 11
        b = self.sum_1 - a               # 10 = (1-count of input_1) - 11
        c = self.sum_2 - a               # 01 = (1-count of input_2) - 11
        n = self.timestep_cur            # run length == number of forward() calls
        d = n - a - b - c                # 00: the pairs accounted for nowhere else
        ad_minus_bc = a * d - b * c
        ad_gt_bc = torch.gt(ad_minus_bc, 0).type(self.ntype)
        ad_le_bc = 1 - ad_gt_bc
        a_plus_b = a + b
        a_plus_c = a + c
        a_minus_d = a - d
        zeros = torch.zeros_like(a)
        ones = torch.ones_like(a)
        # SCC denominator differs by the sign of (ad - bc); max(., 1) guards div-by-0.
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
        assert self.valid, logger.error('Metric is not valid. Please call forward() before analyze().')
        # one property access: correlation computes from the accumulated counts on each read
        correlation = self.correlation
        result = analyze(
            correlation,
            verbose=verbose,
            report='Correlation',
            value='correlation',
            timestep=self.timestep_cur,
        )
        return correlation, result

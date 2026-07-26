import torch

from napl.sim.base import napl_base
from napl.sim.metric._shared import analyze
from loguru import logger


class correlation(napl_base):
    """
    Stochastic cross-correlation (SCC) between two spike streams, accumulated over
    timesteps. Call forward(input_1, input_2) once per timestep to accumulate the joint
    histogram of bit pairs, then analyze() to compute the SCC. If only input_1 is given,
    the SCC is computed between the stream and its one-step-delayed self
    (autocorrelation). Reference: "Exploiting Correlation in Stochastic Circuit
    Design".
    """
    def __init__(
            self,
            config={}
        ):
        super().__init__(config, [])

        # sufficient statistics for the joint bit-pair histogram (a=11, b=10, c=01, d=00):
        # the co-occurrence count and the per-stream 1-counts. b, c, d and the run length
        # are all recovered at report() from these plus timestep_cur, so each step accumulates
        # only three counts and forms one product.
        self.paired_11 = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)
        self.sum_1 = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)
        self.sum_2 = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)
        # one-step delay buffer for the autocorrelation (single-input) case
        self.input_1_d = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)


    def _reset(self):
        for p in [self.paired_11, self.sum_1, self.sum_2, self.input_1_d]:
            p.data = torch.zeros_like(p.data)


    def forward(self, input_1, input_2=None):
        if input_2 is None:
            input_2 = self.input_1_d.clone().detach()
            self.input_1_d.data = input_1.clone().detach().type(self.ntype)

        # bool is left uncast: addcmul/add promote it to the ntype accumulator, so the
        # two per-timestep .type() casts are redundant dispatches (kept as int8/float 0/1).
        input_1_is_1 = torch.ne(input_1, 0)
        input_2_is_1 = torch.ne(input_2, 0)

        # out-of-place add (assigned to .data) so the scalar accumulators broadcast up
        # to the input shape on the first call, matching the decoder/accuracy idiom.
        # addcmul fuses the 11-product and the add, saving one temporary per timestep.
        self.paired_11.data = torch.addcmul(self.paired_11, input_1_is_1, input_2_is_1)
        self.sum_1.data = self.sum_1.add(input_1_is_1)
        self.sum_2.data = self.sum_2.add(input_2_is_1)


    @property
    def correlation(self):
        """
        SCC, computed on access from the accumulated bit-pair counts.
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
        # return the correlation and index of max abs correlation
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
        self.correlation_abs_max = result.absolute_max
        self.correlation_abs_min = result.absolute_min
        self.correlation_avg = result.mean
        self.correlation_mae = result.mean_absolute
        self.correlation_rmse = result.root_mean_square

        return correlation, result.max_absolute_index

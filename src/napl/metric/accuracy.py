import torch

from napl.base import napl_base
from napl.metric._shared import analyze
from napl.utils import *
from loguru import logger


class accuracy(napl_base):
    """
    Calculate progressive accuracy based on progressive precision of input spike.
    Progressive precision: 'Fast and accurate computation using stochastic circuits'
    """
    def __init__(
            self,
            config={
                'polarity' : 'bipolar',
            }
        ):
        super().__init__(config, ['polarity'])

        self.spike_count = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.spike_error = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        # absolute max error
        self.spike_error_abs_max = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # absolute min error
        self.spike_error_abs_min = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # mean error
        self.spike_error_avg = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # mean absolute error
        self.spike_error_mae = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # root mean square error
        self.spike_error_rmse = torch.nn.Parameter(torch.zeros(1), requires_grad=False)


    def reset(self, verbose=False):
        """
        Reset the timestep and one count.
        """
        self.timestep_cur = 0
        self.spike_count.data = torch.zeros(1, device=self.spike_count.device)
        self.spike_error.data = torch.zeros(1, device=self.spike_error.device)
        self.spike_error_abs_max.data = torch.zeros(1, device=self.spike_error_abs_max.device)
        self.spike_error_abs_min.data = torch.zeros(1, device=self.spike_error_abs_min.device)
        self.spike_error_avg.data = torch.zeros(1, device=self.spike_error_avg.device)
        self.spike_error_mae.data = torch.zeros(1, device=self.spike_error_mae.device)
        self.spike_error_rmse.data = torch.zeros(1, device=self.spike_error_rmse.device)


    @property
    def spike_value(self):
        """
        Progressive value, computed on access from the accumulated spike_count.
        Returns a fresh tensor; before any forward() it is the zeros seed.
        """
        if self.timestep_cur == 0:
            return torch.zeros_like(self.spike_count)
        # sv is the fresh div result, so the in-place bipolar rescale leaves spike_count untouched.
        sv = self.spike_count.div(self.timestep_cur)
        if self.polarity == 'bipolar':
            sv.mul_(2).sub_(1)
        return sv


    def forward(self, spike: torch.Tensor):
        # float accumulator avoids overflow; the 0/1 spike promotes exactly, so no cast.
        sc = self.spike_count
        # shape-guarded: first forward broadcasts the (1,) seed up to spike's shape
        # out-of-place; steady state accumulates in place to drop a per-timestep alloc.
        if sc.shape == spike.shape:
            sc.add_(spike)
        else:
            sc.data = sc.add(spike)
        # no return: evaluating the spike_value property here would redo the div
        # every timestep; readers access .spike_value on demand instead.


    def analyze(
        self,
        reference: torch.Tensor,
        verbose=False,
        *,
        scale_ref=1,
    ):
        # return the error and index of max abs error
        assert self.valid, logger.error(f'Metric is not valid. Please call forward() before analyze().')
        # one property access: spike_value computes from spike_count on each read
        spike_value = self.spike_value
        self.spike_error.data = spike_value.sub(reference.div(scale_ref))
        result = analyze(
            self.spike_error,
            verbose=verbose,
            report='Accuracy',
            value='error',
            timestep=self.timestep_cur,
        )
        self.spike_error_abs_max.data = result.absolute_max
        self.spike_error_abs_min.data = result.absolute_min
        self.spike_error_avg.data = result.mean
        self.spike_error_mae.data = result.mean_absolute
        self.spike_error_rmse.data = result.root_mean_square

        return self.spike_error, result.max_absolute_index

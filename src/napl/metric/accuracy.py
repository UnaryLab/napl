import torch

from napl.base import napl_base
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


    def analyze(self, reference: torch.Tensor, verbose=False):
        # return the error and index of max abs error
        assert self.valid, logger.error(f'Metric is not valid. Please call forward() before analyze().')
        # one property access: spike_value computes from spike_count on each read
        spike_value = self.spike_value
        self.spike_error.data = spike_value.sub(reference)
        # abs() is reused 4x below; compute once to avoid redundant full-tensor passes.
        spike_error_abs = self.spike_error.abs()
        # aminmax: one fused pass for both extremes instead of separate min/max scans.
        spike_error_abs_amin, spike_error_abs_amax = torch.aminmax(spike_error_abs)
        self.spike_error_abs_max.data = spike_error_abs_amax
        self.spike_error_abs_min.data = spike_error_abs_amin
        self.spike_error_avg.data = self.spike_error.mean()
        self.spike_error_mae.data = spike_error_abs.mean()
        self.spike_error_rmse.data = torch.sqrt(spike_error_abs.pow(2).mean())

        if verbose:
            logger.info(f'Accuracy report for accuracy instance <{self.name}> over <{self.timestep_cur}> timesteps: ')
            logger.info(f'    Max absolute error:     <{self.spike_error_abs_max.item()}>')
            logger.info(f'    Min absolute error:     <{self.spike_error_abs_min.item()}>')
            logger.info(f'    Mean error:             <{self.spike_error_avg.item()}>')
            logger.info(f'    Mean absolute error:    <{self.spike_error_mae.item()}>')
            logger.info(f'    Root mean square error: <{self.spike_error_rmse.item()}>')
            logger.info(f'')
        return self.spike_error, torch.argmax(spike_error_abs)


def analyze_error(spike_value: torch.Tensor, reference: torch.Tensor):
    # return the error and index of max abs error
    spike_error = spike_value.sub(reference)
    # abs() is reused 4x below; compute once to avoid redundant full-tensor passes.
    spike_error_abs = spike_error.abs()
    # aminmax: one fused pass for both extremes instead of separate min/max scans.
    spike_error_abs_min, spike_error_abs_max = torch.aminmax(spike_error_abs)
    spike_error_avg = spike_error.mean()
    spike_error_mae = spike_error_abs.mean()
    spike_error_rmse = torch.sqrt(spike_error_abs.pow(2).mean())

    logger.info(f'Accuracy report: ')
    logger.info(f'    Max absolute error:     <{spike_error_abs_max.item()}>')
    logger.info(f'    Min absolute error:     <{spike_error_abs_min.item()}>')
    logger.info(f'    Mean error:             <{spike_error_avg.item()}>')
    logger.info(f'    Mean absolute error:    <{spike_error_mae.item()}>')
    logger.info(f'    Root mean square error: <{spike_error_rmse.item()}>')
    logger.info(f'')
    return spike_error, torch.argmax(spike_error_abs)


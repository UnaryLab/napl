import torch

from napl.sim.base import napl_base
from napl.sim.metric._shared import analyze
from loguru import logger


class accuracy(napl_base):
    """
    Track how closely a spike stream's running decoded value matches a reference.

    Use this metric to inspect progressive error as more timesteps arrive or to
    decide whether a stream has reached sufficient precision.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import accuracy

        metric = accuracy()
        metric(torch.tensor([1.0, 0.0]))
        value = metric.spike_value
        error, result = metric.analyze(torch.tensor([1.0, -1.0]))
        metric.reset()

    .. container:: api-references

        .. rubric:: References

        *Fast and accurate computation using stochastic circuits*.
    """


    def __init__(
            self,
            config={
                'polarity' : 'bipolar',
            }
        ):
        """
        Configure how the metric decodes its spike stream.

        If ``config`` is supplied, it must contain ``polarity``.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping. The default is
              ``{'polarity': 'bipolar'}``.

              - **polarity**: Choose ``"unipolar"`` for values in ``[0, 1]`` or ``"bipolar"`` for values in ``[-1, 1]``; the default is ``"bipolar"``.
              - **name**: Optional instance label; the default is ``None``.
        """
        super().__init__(config, ['polarity'], polarity_required=True)

        #: Running count of one-valued spikes for each observed stream element.
        self.spike_count: torch.Tensor
        self.register_buffer('spike_count', torch.zeros(1))


    def _reset(self):
        """
        Clear the class-local spike count and restore its scalar seed shape.

        The inherited :meth:`napl.sim.base.napl_base.reset` method resets the
        timestep before calling this hook. This hook returns ``None``.
        """
        self.spike_count.resize_(1).zero_()


    def forward(self, spike: torch.Tensor):
        """
        Record one timestep of a spike stream.

        Use a 0/1 tensor with the same logical shape at every timestep in one
        measurement. Calling the metric increments its timestep, updates the
        running count, and returns ``None``.

        Args:
            spike: Spike values for the current timestep.

        **Example:**

        .. code-block:: python

            metric(torch.tensor([1.0, 0.0]))
        """
        # A float accumulator avoids overflow, and 0/1 spikes promote exactly.
        sc = self.spike_count
        # The scalar seed broadcasts once; matching shapes then accumulate in place.
        if sc.shape == spike.shape:
            sc.add_(spike)
        else:
            expanded = sc.add(spike).detach()
            self.spike_count.resize_as_(expanded).copy_(expanded)


    @property
    def spike_value(self):
        """
        Return the running decoded value with the spike input's logical shape.

        The result uses ``[0, 1]`` for unipolar streams and ``[-1, 1]`` for
        bipolar streams. Before the first timestep, it returns zero. Reading
        this property does not change the accumulated state.

        **Example:**

        .. code-block:: python

            value = metric.spike_value
        """
        if self.timestep_cur == 0:
            return torch.zeros_like(self.spike_count)
        # Division returns a fresh tensor, so bipolar rescaling cannot alter the count.
        sv = self.spike_count.div(self.timestep_cur)
        if self.polarity == 'bipolar':
            sv.mul_(2).sub_(1)
        return sv


    def analyze(
        self,
        reference: torch.Tensor,
        verbose=False,
        *,
        scale_ref=1,
    ):
        """
        Compare the decoded stream with a numeric reference.

        Call this method after at least one timestep.

        Args:
            reference: Expected numeric value with the same logical shape as the spike stream.
            verbose: Set to ``True`` to print the analysis summary.
            scale_ref: Divisor applied to ``reference``; the comparison uses ``reference / scale_ref``.

        Returns:
            A pair containing the per-element signed progressive error and its complete :class:`napl.sim.metric._shared.Analysis` summary.

        This method does not change the accumulated metric state.

        **Example:**

        .. code-block:: python

            error, result = metric.analyze(torch.tensor([1.0, -1.0]))

        """
        assert self.valid, logger.error('Metric is not valid. Please call forward() before analyze().')
        spike_value = self.spike_value
        progressive_error = spike_value.sub(reference.div(scale_ref)).detach()
        result = analyze(
            progressive_error,
            verbose=verbose,
            report='Progressive Error',
            value='progressive error',
            timestep=self.timestep_cur,
        )
        return progressive_error, result

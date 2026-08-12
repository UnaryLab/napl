import torch

from loguru import logger

from napl.sim.base import napl_base


class eq_rc(napl_base):
    r"""
    Compare two rate-coded streams for an equal-within-tolerance result.

    This is a *derived* operation: the equality counterpart of
    :class:`~napl.sim.operation.gt_rc` and :class:`~napl.sim.operation.lt_rc`,
    which decide the strict order of two running rates. ``eq_rc`` instead
    decides whether the two running rates coincide to within a rate tolerance,
    so the three together cover the ordering of a rate-coded pair.

    Use it to gate on approximate rate equality: a one output marks the two
    streams as carrying the same decoded value up to ``tolerance``.

    .. rubric:: Decision rule

    Let :math:`c_t=\sum_{k\le t}(x_{0,k}-x_{1,k})` be the running spike-count
    difference after timestep :math:`t`, and let :math:`p_i(t)` be the running
    decoded rate of stream :math:`i`. The output at timestep :math:`t` is

    .. math::

       y_t = \mathbf{1}\{\,|p_0(t)-p_1(t)| \le \texttt{tolerance}\,\}
           = \mathbf{1}\{\,g\,|c_t| \le \texttt{tolerance}\cdot t\,\},

    where :math:`g=1` for a unipolar code (:math:`p=c/t`) and :math:`g=2` for a
    bipolar code (:math:`p=2c/t-1`), the value-domain gain of the rate code. The
    comparison is combinational on the running count register, so the output at
    timestep :math:`t` already reflects that timestep's spikes.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import eq_rc

        compare = eq_rc({'polarity': 'unipolar', 'tolerance': 0.1})
        result = compare(torch.tensor([1], dtype=torch.int8),
                         torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        Derived from the rate comparators
        *:class:`~napl.sim.operation.gt_rc`* and
        *:class:`~napl.sim.operation.lt_rc`*.
    """


    def __init__(
            self,
            config={
                'tolerance' : 0.05,
            }
    ):
        """
        Construct the comparator with its running rate-difference counter.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **tolerance**: Rate-equality band in decoded value units; the
                output is one while the two running rates differ by no more than
                this. A real number in ``[0, 2]``; the default is ``0.05``.
              - **polarity**: Optional stream polarity, ``"unipolar"`` or
                ``"bipolar"``. It sets the value-domain gain of the rate code
                (``2`` for bipolar, ``1`` otherwise); absent, unipolar is
                assumed.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['tolerance'], optional_key_list=['polarity'], polarity_required=False)

        #: Rate-equality band in decoded value units.
        self.tolerance = config['tolerance']
        if not isinstance(self.tolerance, (int, float)) or not 0.0 <= self.tolerance <= 2.0:
            message = f'Invalid tolerance: <{self.tolerance}>; expected a real number in [0, 2].'
            logger.error(message)
            raise AssertionError(message)
        # Bipolar rate p = 2c/t - 1, so a value-domain gap is twice the count-rate gap.
        #: Value-domain gain of the rate code used to scale the count threshold.
        self.value_gain = 2.0 if self.polarity == 'bipolar' else 1.0
        #: Running spike-count difference carried across timesteps.
        self.count: torch.Tensor
        self.register_buffer('count', torch.zeros(1, dtype=self.ntype))
        #: Whether :attr:`count` must be expanded for the first input shape.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the combinational comparator.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'output': 'unipolar'}
        self.correlation_i = {}


    def _reset(self):
        """
        Clear the local running count and first-call shape state.
        """
        self.count.resize_(1).zero_()
        self.is_first_call = True


    def forward(self, input_0, input_1):
        """
        Compare one timestep from two rate-coded streams.

        Args:
            input_0: Current 0/1 spikes from the first stream.
            input_1: Current 0/1 spikes from the second stream.

        Returns:
            A ``1`` spike where the two running rates are equal to within
            ``tolerance`` after this timestep, else ``0``. The call updates the
            running count difference.

        **Example:**

        .. code-block:: python

            result = compare(torch.tensor([1], dtype=torch.int8),
                             torch.tensor([1], dtype=torch.int8))
        """
        diff = (input_0 - input_1).type(self.ntype)
        if self.is_first_call:
            self.count.resize_as_(diff).zero_()
            self.is_first_call = False
        self.count.add_(diff)

        # y_t = 1{ value-gap <= tolerance }; value-gap = value_gain * |count| / t.
        threshold = self.tolerance / self.value_gain * self.timestep_cur
        output = self.count.abs().le(threshold)
        return output.type(self.stype)

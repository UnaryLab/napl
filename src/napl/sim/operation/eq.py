import torch

from loguru import logger

from napl.sim.base import napl_base


class eq(napl_base):
    r"""
    Compare two rate-coded streams with a saturating up/down counter.

    This is a *derived* operation: the equality counterpart of
    :class:`~napl.sim.operation.gt` and :class:`~napl.sim.operation.lt`,
    which decide the strict order of two rate-coded streams. ``eq`` instead
    decides whether the two streams carry the same value, so the three together
    cover the ordering of a rate-coded pair.

    Use it to gate on approximate equality: a one output marks the two streams
    as still balanced within the counter band.

    .. rubric:: Decision rule

    A single unsigned **width**-bit counter starts at :math:`h=2^{width-1}` and
    steps once per timestep from the input spike pair, saturating at :math:`0`
    and :math:`2^{width}-1`:

    .. math::

       c_t = \mathrm{sat}\!\left(c_{t-1} + x_{0,t} - x_{1,t}\right),\qquad
       y_t = \mathbf{1}\{\,|c_t - h| \le \texttt{tolerance}\,\}.

    A ``10`` input pair counts up, ``01`` counts down, and ``00`` and ``11``
    hold. The counter therefore tracks the running spike-count difference of the
    two streams, bounded by its own saturation, and a persistent rate gap of
    either sign drives it onto a rail and holds :math:`y` at zero. The output is
    combinational on the post-update counter, so the output at timestep
    :math:`t` already reflects that timestep's spikes. The rate difference of a
    rate code is proportional to the spike-count difference in both the unipolar
    and the bipolar code, so the circuit does not depend on polarity.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import eq

        compare = eq({'width': 3, 'tolerance': 1})
        result = compare(torch.tensor([1], dtype=torch.int8),
                         torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        Derived from the rate comparators
        *:class:`~napl.sim.operation.gt`* and
        *:class:`~napl.sim.operation.lt`*.
    """
    #: Dominant hardware mechanism of this class.
    mechanism = 'finite-state-machine'


    def __init__(
            self,
            config={
                'width' : 3,
                'tolerance' : 1,
            }
    ):
        """
        Construct the comparator with its saturating counter.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Counter bit width, giving ``2**width`` states; an
                integer of at least ``1``. The default is ``3``.
              - **tolerance**: Half-width of the equality band around the
                counter's half-scale state, in counter steps; an integer in
                ``[0, 2**(width-1)]``. The default is ``1``.
              - **polarity**: Optional stream polarity, ``"unipolar"`` or
                ``"bipolar"``. It does not change the circuit; absent, unipolar
                is assumed.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['width', 'tolerance'], optional_key_list=['polarity'], polarity_required=False)

        #: Counter bit width, giving ``2**width`` counter states.
        self.width = config['width']
        if not isinstance(self.width, int) or isinstance(self.width, bool) or self.width < 1:
            message = f'Invalid width: <{self.width}>; expected an integer of at least 1.'
            logger.error(message)
            raise AssertionError(message)

        #: Largest value retained by the saturating counter.
        self.cnt_max = 2**self.width - 1
        #: Half-scale counter value restored by ``_reset``, the balanced state.
        self.cnt_half = 2**(self.width - 1)

        #: Half-width of the equality band around ``cnt_half``, in counter steps.
        self.tolerance = config['tolerance']
        if (not isinstance(self.tolerance, int) or isinstance(self.tolerance, bool)
                or not 0 <= self.tolerance <= self.cnt_half):
            message = (f'Invalid tolerance: <{self.tolerance}>; expected an integer in '
                       f'[0, {self.cnt_half}].')
            logger.error(message)
            raise AssertionError(message)

        # The scalar initial counter broadcasts to the input shape on first use.
        #: Saturating counter tracking the spike-count difference of the two streams.
        self.cnt: torch.Tensor
        # The default design point is width 3 with tolerance 1.
        self.register_buffer('cnt', torch.zeros(1, dtype=self.ntype).fill_(self.cnt_half))
        #: Hardware latency and timing metadata for the combinational comparator.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'output': 'unipolar'}
        self.correlation_i = {}


    def _reset(self):
        """
        Restore the counter to its half-scale balanced state.
        """
        self.cnt.resize_(1).fill_(self.cnt_half)


    def forward(self, input_0, input_1):
        """
        Compare one timestep from two rate-coded streams.

        Args:
            input_0: Current 0/1 spikes from the first stream.
            input_1: Current 0/1 spikes from the second stream.

        Returns:
            A ``1`` spike where the counter lies within ``tolerance`` of its
            half-scale state after this timestep, else ``0``. The call steps and
            saturates the counter.

        **Example:**

        .. code-block:: python

            result = compare(torch.tensor([1], dtype=torch.int8),
                             torch.tensor([1], dtype=torch.int8))
        """
        # A 10 pair steps up, 01 steps down, and 00 and 11 leave the difference unchanged.
        step = (input_0 - input_1).type(self.ntype)
        if self.cnt.shape == step.shape:
            self.cnt.add_(step).clamp_(0, self.cnt_max)
        else:
            updated = self.cnt.add(step).clamp_(0, self.cnt_max)
            self.cnt.resize_as_(updated).copy_(updated.detach())
        output = self.cnt.sub(self.cnt_half).abs_().le(self.tolerance)
        return output.type(self.stype)

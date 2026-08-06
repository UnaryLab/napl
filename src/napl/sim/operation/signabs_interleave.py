import torch

from loguru import logger
from napl.sim.base import hw_params, napl_base


class signabs_interleave(napl_base):
    r"""
    Split bipolar rate-coded spikes into sign and interleaved magnitude streams.

    Use this streaming counter-based kernel when the magnitude stream should be
    derived from alternating counter parity rather than directly from each input bit.

    The target splits a bipolar value :math:`v` into its sign and magnitude, so
    that the returned streams decode to

    .. math::

       \mathrm{sign} = \mathbf{1}\{v < 0\},\qquad
       \mathrm{magnitude} = |v|.

    A saturating counter of **width** bits estimates the sign online, and the
    magnitude comes from the counter parity rather than from the current input
    spike. The parity interleaves the magnitude spikes across timesteps, which
    decorrelates them from the input stream at the cost of tracking a changing
    input more slowly.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import signabs_interleave

        operation = signabs_interleave({'width': 3})
        sign, magnitude = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*, IEEE Design & Test, 2021.
    """


    def __init__(self, config={'width': 3}):
        """
        Configure the interleaving counter.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Positive saturating-counter bit width; the default is ``3``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['width'], optional_key_list=['polarity'], polarity_required=False)

        #: Width of the bounded sign-and-magnitude accumulator in bits.
        self.width = config['width']
        if not isinstance(self.width, int) or self.width <= 0:
            message = f'Invalid width: <{self.width}>; legal values: a positive integer.'
            logger.error(message)
            raise AssertionError(message)
        #: Largest value retained by the unsigned accumulator.
        self.acc_max = 2**self.width - 1
        #: Half-scale accumulator value that represents bipolar zero.
        self.acc_half = 2 ** (self.width - 1)
        #: Running bipolar input count used to interleave sign and magnitude spikes.
        self.acc: torch.Tensor
        self.register_buffer('acc',
            torch.full((1,), self.acc_half, dtype=self.ntype),
        )
        #: Hardware latency and timing metadata for the combinational outputs.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'rc', 'magnitude': 'rc'}
        self.polarity_io = {'input': 'bipolar', 'magnitude': 'unipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the interleaving accumulator to its half-scale initial state.
        """
        self.acc.resize_(1).fill_(self.acc_half)


    def forward(self, input: torch.Tensor):
        """
        Process one timestep of a bipolar rate-coded stream.

        The call updates and saturates the accumulator, estimates the sign, and
        forms the magnitude spike from the sign and accumulator parity.

        Args:
            input: Tensor of current 0/1 bipolar input spikes.

        Returns:
            Pair ``(sign, magnitude)`` of 0/1 spike tensors matching ``input``.

        **Example:**

        .. code-block:: python

            sign, magnitude = operation(torch.tensor([0.0, 1.0]))
        """
        if self.acc.shape == input.shape:
            self.acc.add_(input, alpha=2).sub_(1).clamp_(0, self.acc_max)
        else:
            updated = self.acc.add(input, alpha=2).sub_(1).clamp_(0, self.acc_max)
            self.acc.resize_as_(updated).copy_(updated.detach())

        sign = torch.lt(self.acc, self.acc_half).type(torch.int8)
        acc_odd = self.acc.remainder(2).type(torch.int8)
        magnitude = sign ^ acc_odd
        return sign.type(self.stype), magnitude.type(self.stype)

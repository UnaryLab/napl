import torch

from loguru import logger
from napl.sim.base import hw_params, napl_base


class signabs_interleave(napl_base):
    """
    Split bipolar rate-coded spikes into sign and interleaved magnitude streams.

    Use this streaming counter-based kernel when the magnitude stream should be
    derived from alternating counter parity rather than directly from each input bit.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import signabs_interleave

        operation = signabs_interleave({'width': 5})
        sign, magnitude = operation(torch.tensor([0.0, 1.0]))
    """


    def __init__(self, config={'width': 5}):
        """
        Configure the interleaving counter.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Positive saturating-counter bit width; the default is ``5``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['width'], polarity_required=False)

        #: Width of the bounded sign-and-magnitude accumulator in bits.
        self.width = config['width']
        assert isinstance(self.width, int) and self.width > 0, logger.error(
            f'Invalid width: <{self.width}>; legal values: a positive integer.'
        )
        #: Largest value retained by the unsigned accumulator.
        self.acc_max = 2**self.width - 1
        #: Half-scale accumulator value that represents bipolar zero.
        self.acc_half = 2 ** (self.width - 1)
        #: Hardware latency and timing metadata for the combinational outputs.
        self.hw = hw_params(pp_delay=0)
        #: Running bipolar input count used to interleave sign and magnitude spikes.
        self.acc: torch.Tensor
        self.register_buffer('acc',
            torch.full((1,), self.acc_half, dtype=self.ntype),
        )


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
            self.acc.add_(input, alpha=2).sub_(1).clamp_(
                0, self.acc_max
            )
        else:
            updated = self.acc.add(input, alpha=2).sub_(1).clamp_(
                0, self.acc_max
            )
            self.acc.resize_as_(updated).copy_(updated.detach())

        sign = torch.lt(self.acc, self.acc_half).type(torch.int8)
        acc_odd = self.acc.remainder(2).type(torch.int8)
        magnitude = sign ^ acc_odd
        return sign.type(self.stype), magnitude.type(self.stype)

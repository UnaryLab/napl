import torch

from napl.utils import *
from napl.sim.base import napl_base, hw_params


class signabs(napl_base):
    """
    Split a bipolar rate-coded stream into sign and magnitude streams.

    Use this streaming counter-based kernel when downstream unsigned operations
    need a magnitude stream plus a sign stream. A sign bit of ``0`` denotes
    non-negative and ``1`` denotes negative.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import signabs

        operation = signabs({'width': 3})
        sign, magnitude = operation(torch.tensor([0.0, 1.0]))
    """


    def __init__(
            self,
            config={
                'width' : 3,
            }
    ):
        """
        Configure the sign-estimation counter.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Saturating-counter bit width; the default is ``3``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['width'], polarity_required=False)
        #: Hardware latency and timing metadata for the combinational outputs.
        self.hw = hw_params(pp_delay=0)

        #: Width of the bounded sign-and-magnitude accumulator in bits.
        self.width = config['width']

        #: Largest value retained by the unsigned accumulator.
        self.acc_max = 2**self.width - 1
        #: Half-scale accumulator value that represents bipolar zero.
        self.acc_med = 2**(self.width - 1)
        #: Running bipolar input count used to derive sign and magnitude spikes.
        self.acc: torch.Tensor
        self.register_buffer('acc', torch.zeros(1, dtype=self.ntype).fill_(self.acc_med))


    def _reset(self):
        """
        Restore the sign accumulator to its half-scale initial state.
        """
        self.acc.resize_(1).fill_(self.acc_med)


    def forward(self, input):
        """
        Process one timestep of a bipolar rate-coded stream.

        The call updates and saturates the sign accumulator, then derives the
        magnitude spike by XORing the current input with the estimated sign.

        Args:
            input: Tensor of current 0/1 bipolar input spikes.

        Returns:
            Pair ``(sign, magnitude)`` of 0/1 spike tensors matching ``input``.

        **Example:**

        .. code-block:: python

            sign, magnitude = operation(torch.tensor([0.0, 1.0]))
        """
        # The scalar initial accumulator broadcasts out of place; matching shapes update in place.
        if self.acc.shape == input.shape:
            self.acc.add_(input, alpha=2).sub_(1).clamp_(0, self.acc_max)
        else:
            updated = self.acc.add(input, alpha=2).sub_(1).clamp_(0, self.acc_max)
            self.acc.resize_as_(updated).copy_(updated.detach())
        sign = torch.lt(self.acc, self.acc_med).type(torch.int8)
        abs = sign ^ input.type(torch.int8)
        return sign.type(self.stype), abs.type(self.stype)

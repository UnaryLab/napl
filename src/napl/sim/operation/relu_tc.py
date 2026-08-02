import torch

from loguru import logger
from napl.sim.base import hw_params, napl_base


class relu_tc(napl_base):
    """
    Apply ReLU to a bipolar temporal-coded stream.

    Use this streaming kernel with temporal 0/1 codes whose configured width
    determines the half-scale phase boundary.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import relu_tc

        operation = relu_tc({'width': 8})
        output = operation(torch.tensor([0.0, 1.0]))
    """


    def __init__(self, config={'width': 8}):
        """
        Configure the temporal-code width.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Positive temporal-code bit width; the default is ``8``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['width'], polarity_required=False)

        #: Number of temporal-code bits in one input value.
        self.width = config['width']
        assert isinstance(self.width, int) and self.width > 0, logger.error(
            f'Invalid width: <{self.width}>; legal values: a positive integer.'
        )
        #: Midpoint cycle that separates the two temporal output phases.
        self.threshold = 2 ** (self.width - 1)
        #: Hardware latency and timing metadata for the combinational output path.
        self.hw = hw_params(pp_delay=0)
        #: Running sum of input bits in the current temporal codeword.
        self.acc: torch.Tensor
        self.register_buffer('acc', torch.zeros(1, dtype=self.ntype))
        #: Number of temporal-code bits processed since reset.
        self.cycle = 0


    def _reset(self):
        """
        Clear the temporal accumulator and restart the local cycle counter.
        """
        self.acc.resize_(1).zero_()
        self.cycle = 0


    def forward(self, input: torch.Tensor):
        """
        Process one timestep of a bipolar temporal code.

        The call increments the local cycle, accumulates ``input``, and emits
        the phase-dependent ReLU bit.

        Args:
            input: Tensor of current 0/1 temporal-code bits.

        Returns:
            0/1 output tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        self.cycle += 1
        if self.acc.shape == input.shape:
            self.acc.add_(input)
        else:
            updated = self.acc.add(input)
            self.acc.resize_as_(updated).copy_(updated.detach())

        if self.cycle <= self.threshold:
            output = torch.le(self.acc, self.cycle)
        else:
            output = torch.gt(self.acc, self.threshold) & input.type(
                torch.bool
            )
        return output.type(self.stype)

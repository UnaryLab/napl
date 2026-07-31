import torch
from collections import deque

from napl.utils import *
from napl.sim.base import napl_base, hw_params


class shiftreg(napl_base):
    """
    Delay a spike tensor with an alternating-initialized shift register.

    Use this streaming register where the initial delay contents should alternate
    between ``0`` and ``1`` instead of starting at zero. After filling, each call
    returns the input from ``depth`` timesteps earlier.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import shiftreg

        delay = shiftreg({'depth': 2})
        output = delay(torch.tensor([1], dtype=torch.int8))
    """
    def __init__(
            self,
            config={'depth': 1}
        ):
        """
        Configure the register depth.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **depth**: Number of stored timesteps and output delay cycles; the default is ``1``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['depth'], polarity_required=False)

        self.depth = config['depth']
        # output is reg[head], which is depth cycles old: latency == depth. RTL
        # i_rst_n must reproduce the i%2 reset pattern below, not all-zeros.
        self.hw = hw_params(pp_delay=self.depth)
        self.register_buffer('reg', torch.zeros(self.depth, dtype=self.stype))
        for i in range(self.depth):
            self.reg[i].fill_(i%2)
        self.is_first_call = True
        # runtime FIFO of row tensors; built lazily from self.reg on first forward
        self.fifo = None


    def _reset(self):
        """
        Restore alternating register contents and discard the runtime queue.
        """
        self.reg.resize_(self.depth).zero_()
        for i in range(self.depth):
            self.reg[i].fill_(i%2)
        self.is_first_call = True
        self.fifo = None


    def forward(self, input: torch.tensor):
        """
        Push one input timestep through the shift register.

        Args:
            input: Current spike tensor.

        Returns:
            The oldest stored tensor. Initial calls return the alternating reset
            pattern; later calls return prior inputs. The current input is copied
            into the queue.

        **Example:**

        .. code-block:: python

            output = delay(torch.tensor([1], dtype=torch.int8))
        """
        # input is a spike tensor
        if self.is_first_call:
            input_shape = list(input.shape)
            input_shape.insert(0, self.depth)
            self.reg.resize_(input_shape).zero_()
            for i in range(self.depth):
                self.reg[i].fill_(i%2)
            # FIFO over row tensors, oldest at the left.
            self.fifo = deque(self.reg[i] for i in range(self.depth))
            self.is_first_call = False

        # Read the depth-cycles-old value and snapshot the new input.
        output = self.fifo.popleft()
        self.fifo.append(input.detach().clone())
        return output

import torch

from loguru import logger
from napl.sim.base import napl_base


class relu_shiftreg(napl_base):
    r"""
    Apply ReLU to a bipolar rate-coded stream using a shift-register estimate.

    The target rate-domain operation is

    .. math::

       y = \max(x,0).

    A **depth**-entry shift register holds the recent output spikes, and the
    kernel forces extra one-spikes whenever their count falls below half the
    register depth, so the result approximates the target to the resolution of
    the register. The input and the output are both bipolar 0/1 spike streams.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import relu_shiftreg

        operation = relu_shiftreg({'depth': 4})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """


    def __init__(self, config={'depth': 4}):
        """
        Configure the shift-register estimator.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **depth**: Shift-register length as an integer in ``[1, 127]``; the default is ``4``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['depth'], optional_key_list=['polarity'], polarity_required=False)

        #: Number of spike-history entries retained by the ReLU register.
        self.depth = config['depth']
        if not isinstance(self.depth, int) or not (0 < self.depth <= 127):
            message = f'Invalid depth: <{self.depth}>; legal values: integers in [1, 127].'
            logger.error(message)
            raise AssertionError(message)
        #: Half-depth count threshold that represents bipolar zero.
        self.depth_half = self.depth / 2

        #: Circular register of recent ReLU output spikes.
        self.reg: torch.Tensor
        self.register_buffer('reg',
            torch.tensor(
                [index % 2 for index in range(self.depth)], dtype=self.stype
            ),
        )
        #: Number of one-spikes currently stored in :attr:`reg`.
        self.count: torch.Tensor
        self.register_buffer('count', torch.zeros(1, dtype=torch.long))
        #: Previous timestep's register count used by the output decision.
        self.count_delayed: torch.Tensor
        self.register_buffer('count_delayed', torch.zeros(1, dtype=torch.long))
        #: Whether register state must be expanded for the first input shape.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the combinational output path.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'bipolar', 'output': 'bipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the alternating register seed, counters, and first-call flag.
        """
        self.reg.resize_(self.depth)
        for index in range(self.depth):
            self.reg[index].fill_(index % 2)
        self.count.resize_(1).zero_()
        self.count_delayed.resize_(1).zero_()
        self.is_first_call = True


    def forward(self, input: torch.Tensor):
        """
        Process one timestep of a bipolar rate-coded stream.

        The first call expands and seeds the register for the input shape. Each
        call then replaces the register entry selected by ``timestep_cur`` and
        updates the running count.

        Args:
            input: Tensor of current 0/1 input spikes.

        Returns:
            Bipolar 0/1 ReLU spike tensor with the same shape as ``input``.
            The first call after construction or :meth:`reset` returns all
            ones.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        input_i8 = input.type(torch.int8)
        if self.is_first_call:
            self.reg.resize_((self.depth, *input.shape))
            for index in range(self.depth):
                self.reg[index].fill_(index % 2)
            count = self.reg.sum(dim=0)
            self.count.resize_as_(count).copy_(count.detach())
            output = torch.ones_like(input, dtype=self.stype)
            self.is_first_call = False
        else:
            output = (
                torch.lt(self.count_delayed, self.depth_half) | input_i8
            ).type(self.stype)

        if self.count_delayed.shape == self.count.shape:
            self.count_delayed.copy_(self.count.detach())
        else:
            self.count_delayed.resize_as_(self.count).copy_(self.count.detach())
        head = (self.timestep_cur - 1) % self.depth
        removed = self.reg[head].clone()
        self.reg[head].copy_(output.detach())
        self.count.add_(output).sub_(removed)
        return output

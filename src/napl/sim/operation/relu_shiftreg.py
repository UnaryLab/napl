import torch

from loguru import logger
from napl.sim.base import hw_params, napl_base


class relu_shiftreg(napl_base):
    r"""
    Apply ReLU to a bipolar rate-coded stream using a shift-register estimate.

    The precise target rate-domain operation is

    .. math::

       y = \max(x,0).

    Let R_t be the depth-element register, h_t its circular head, c_t the
    current register count, and d_t the delayed count used by the output
    decision. After first-call initialization with R_0[j] = j mod 2, the
    exact state update is

    .. math::

       \begin{aligned}
       y_0 &= 1,\\
       y_t &= x_t \mathbin{\lor}
       \mathbf{1}\{d_t<depth/2\}\quad (t\geq 1),\\
       c_{t+1} &= c_t+y_t-R_t[h_t],\qquad d_{t+1}=c_t,\\
       R_{t+1}[h_t] &= y_t,\qquad
       h_{t+1}=(h_t+1)\bmod depth.
       \end{aligned}

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import relu_shiftreg

        operation = relu_shiftreg({'depth': 8})
        output = operation(torch.tensor([0.0, 1.0]))
    """


    def __init__(self, config={'depth': 4}):
        """
        Configure the shift-register estimator.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **depth**: Shift-register length as an integer in ``[1, 127]``; the default is ``8``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['depth'], polarity_required=False)

        #: Number of spike-history entries retained by the ReLU register.
        self.depth = config['depth']
        assert isinstance(self.depth, int) and 0 < self.depth <= 127, (
            logger.error(
                f'Invalid depth: <{self.depth}>; legal values: integers in [1, 127].'
            )
        )
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
        #: Circular index of the register row replaced on the next call.
        self.head = 0
        #: Whether register state must be expanded for the first input shape.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the combinational output path.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'bipolar', 'output': 'bipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the alternating register seed, counters, head, and first-call flag.
        """
        self.reg.resize_(self.depth)
        for index in range(self.depth):
            self.reg[index].fill_(index % 2)
        self.count.resize_(1).zero_()
        self.count_delayed.resize_(1).zero_()
        self.head = 0
        self.is_first_call = True


    def forward(self, input: torch.Tensor):
        """
        Process one timestep of a bipolar rate-coded stream.

        The first call expands and seeds the register for the input shape. Each
        call then replaces the oldest register entry, updates the running count,
        and advances the circular head.

        Args:
            input: Tensor of current 0/1 input spikes.

        Returns:
            Bipolar 0/1 ReLU spike tensor with the same shape as ``input``.

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
        removed = self.reg[self.head].clone()
        self.reg[self.head].copy_(output.detach())
        self.count.add_(output).sub_(removed)
        self.head = (self.head + 1) % self.depth
        return output

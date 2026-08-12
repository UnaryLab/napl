import torch

from loguru import logger
from napl.sim.base import napl_base


class signabs_shiftreg(napl_base):
    r"""
    Split bipolar rate-coded spikes using a shift-register sign estimate.

    Use this streaming kernel when recent input history should estimate the sign
    without a multi-bit saturating counter.

    The target splits a bipolar value :math:`v` into its sign and magnitude, so
    that the returned streams decode to

    .. math::

       \mathrm{sign} = \mathbf{1}\{v < 0\},\qquad
       \mathrm{magnitude} = |v|.

    The sign is estimated online from the ones count of the previous **depth**
    inputs, held in a shift register seeded with alternating bits so its initial
    count is balanced. A fixed window tracks a changing input faster than a
    saturating counter but gives a noisier estimate near :math:`v = 0`.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import signabs_shiftreg

        operation = signabs_shiftreg({'depth': 8})
        sign, magnitude = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*, IEEE Design & Test, 2021.
    """


    def __init__(self, config={'depth': 8}):
        """
        Configure the sign-estimation shift register.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **depth**: Shift-register length as an integer in ``[1, 127]``; the default is ``8``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['depth'], optional_key_list=['polarity'], polarity_required=False)

        #: Number of bipolar spike-history entries retained by the converter.
        self.depth = config['depth']
        if not isinstance(self.depth, int) or not (0 < self.depth <= 127):
            message = f'Invalid depth: <{self.depth}>; legal values: integers in [1, 127].'
            logger.error(message)
            raise AssertionError(message)
        #: Half-depth count threshold that represents bipolar zero.
        self.depth_half = self.depth / 2

        #: Circular register of recent bipolar input spikes.
        self.reg: torch.Tensor
        self.register_buffer('reg',
            torch.tensor(
                [index % 2 for index in range(self.depth)], dtype=self.stype
            ),
        )
        #: Number of one-spikes currently stored in :attr:`reg`.
        self.count: torch.Tensor
        self.register_buffer('count', torch.zeros(1, dtype=torch.long))
        #: Whether register state must be expanded for the first input shape.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the combinational outputs.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'sign': 'rc', 'magnitude': 'rc'}
        self.polarity_io = {'input': 'bipolar', 'sign': 'unipolar', 'magnitude': 'unipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the alternating register seed, count, and first-call flag.
        """
        self.reg.resize_(self.depth)
        for index in range(self.depth):
            self.reg[index].fill_(index % 2)
        self.count.resize_(1).zero_()
        self.is_first_call = True


    def forward(self, input: torch.Tensor):
        """
        Process one timestep of a bipolar rate-coded stream.

        The sign is computed from the register count before the current input is
        inserted. The call then replaces the register entry selected by
        ``timestep_cur`` and updates the count.

        Args:
            input: Tensor of current 0/1 bipolar input spikes.

        Returns:
            Pair ``(sign, magnitude)`` of 0/1 spike tensors matching ``input``.

        **Example:**

        .. code-block:: python

            sign, magnitude = operation(torch.tensor([0.0, 1.0]))
        """
        input_stype = input.type(self.stype)
        if self.is_first_call:
            self.reg.resize_((self.depth, *input.shape))
            for index in range(self.depth):
                self.reg[index].fill_(index % 2)
            count = self.reg.sum(dim=0)
            self.count.resize_as_(count).copy_(count.detach())
            self.is_first_call = False

        sign = torch.lt(self.count, self.depth_half).type(torch.int8)
        magnitude = sign ^ input.type(torch.int8)

        head = (self.timestep_cur - 1) % self.depth
        removed = self.reg[head].clone()
        self.reg[head].copy_(input_stype.detach())
        self.count.add_(input_stype).sub_(removed)

        return sign.type(self.stype), magnitude.type(self.stype)

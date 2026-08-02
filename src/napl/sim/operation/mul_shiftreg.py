import torch

from loguru import logger
from napl.sim.base import hw_params, napl_base
from napl.sim.module.encoder import gen_num_seq


class mul_shiftreg(napl_base):
    """
    Multiply unary streams with shift-register decorrelation.

    Use this streaming multiplier when the operand streams may be correlated.
    It stores recent ``input_1`` spikes, samples their population with a number
    sequence, and gates that sample with ``input_0``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mul_shiftreg

        multiply = mul_shiftreg({'polarity': 'unipolar', 'width': 2,
                                 'generator': 'sobol'})
        output = multiply(torch.tensor([1], dtype=torch.int8),
                          torch.tensor([1], dtype=torch.int8))
    """


    def __init__(
        self,
        config={
            'polarity': 'bipolar',
            'width': 4,
            'generator': 'sobol',
        },
    ):
        """
        Configure the decorrelation register and sampling sequence.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **width**: Positive register-address width. The register depth is ``2**width``; the default is ``4``.
              - **generator**: Number-sequence generator name; the default is ``"sobol"``.
              - **dim**: Optional generator dimension; the default used by the implementation is ``1``.
              - **name**: Optional instance label.
        """
        super().__init__(
            config, ['polarity', 'width', 'generator'], polarity_required=True
        )

        #: Address width of the random sequence and shift register.
        self.width = config['width']
        assert isinstance(self.width, int) and self.width > 0, logger.error(
            f'Invalid width: <{self.width}>; legal values: a positive integer.'
        )
        #: Number of random-sequence values and shift-register entries.
        self.depth = 2**self.width
        #: Hardware latency and timing metadata for the combinational output path.
        self.hw = hw_params(pp_delay=0)

        rng_config = {
            'width': self.width,
            'generator': config['generator'],
            'dim': config.get('dim', 1),
        }
        #: Periodic stochastic comparison levels over the register-count range.
        self.rng_seq: torch.Tensor
        self.register_buffer('rng_seq',
            torch.floor(gen_num_seq(rng_config).mul(self.depth)),
        )
        #: Per-element random-sequence index for the direct input path.
        self.rng_idx: torch.Tensor
        self.register_buffer('rng_idx', torch.zeros(1, dtype=torch.long))
        if self.polarity == 'bipolar':
            #: Per-element random-sequence index for the complemented input path.
            self.rng_idx_inv: torch.Tensor
            self.register_buffer('rng_idx_inv', torch.zeros(1, dtype=torch.long))

        #: Circular register of recent spikes from the first multiplicand.
        self.reg: torch.Tensor
        self.register_buffer('reg',
            torch.tensor(
                [index % 2 for index in range(self.depth)], dtype=self.stype
            ),
        )
        #: Per-element number of one-spikes currently stored in :attr:`reg`.
        self.count: torch.Tensor
        self.register_buffer('count', torch.zeros(1, dtype=torch.long))
        #: Circular index of the register row replaced on the next call.
        self.head = 0
        #: Whether the register and index tensors must be expanded for the input shape.
        self.is_first_call = True


    def _reset(self):
        """
        Restore the alternating register contents and restart sequence state.
        """
        self.rng_idx.resize_(1).zero_()
        if self.polarity == 'bipolar':
            self.rng_idx_inv.resize_(1).zero_()
        self.reg.resize_(self.depth)
        for index in range(self.depth):
            self.reg[index].fill_(index % 2)
        self.count.resize_(1).zero_()
        self.head = 0
        self.is_first_call = True


    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor):
        """
        Generate one decorrelated product-spike timestep.

        Args:
            input_0: Current 0/1 spike tensor used to gate and advance sampling.
            input_1: Current 0/1 spike tensor inserted into the shift register.

        Returns:
            A product spike tensor with the broadcast input shape. The call
            advances enabled sequence indices and replaces the oldest stored
            ``input_1`` sample.

        **Example:**

        .. code-block:: python

            output = multiply(torch.tensor([1], dtype=torch.int8),
                              torch.tensor([1], dtype=torch.int8))
        """
        input_0_i8 = input_0.type(torch.int8)
        input_1_stype = input_1.type(self.stype)

        if self.is_first_call:
            self.reg.resize_((self.depth, *input_1.shape))
            for index in range(self.depth):
                self.reg[index].fill_(index % 2)
            count = self.reg.sum(dim=0)
            self.count.resize_as_(count).copy_(count.detach())
            self.is_first_call = False

        source = self.count
        path = input_0_i8 & torch.gt(
            source, self.rng_seq[self.rng_idx]
        )
        if self.rng_idx.shape == input_0_i8.shape:
            self.rng_idx.add_(input_0_i8).remainder_(self.depth)
        else:
            rng_idx = self.rng_idx.add(input_0_i8).remainder_(self.depth)
            self.rng_idx.resize_as_(rng_idx).copy_(rng_idx.detach())

        if self.polarity == 'unipolar':
            output = path
        else:
            input_0_inv = input_0_i8 ^ 1
            path_inv = input_0_inv & ~torch.gt(
                source, self.rng_seq[self.rng_idx_inv]
            )
            if self.rng_idx_inv.shape == input_0_inv.shape:
                self.rng_idx_inv.add_(input_0_inv).remainder_(self.depth)
            else:
                rng_idx_inv = self.rng_idx_inv.add(input_0_inv).remainder_(self.depth)
                self.rng_idx_inv.resize_as_(rng_idx_inv).copy_(rng_idx_inv.detach())
            output = path | path_inv

        removed = self.reg[self.head].clone()
        self.reg[self.head].copy_(input_1_stype.detach())
        self.count.add_(input_1_stype).sub_(removed)
        self.head = (self.head + 1) % self.depth

        return output.type(self.stype)

import torch
from loguru import logger

from napl.sim.base import napl_base, hw_params


class shiftreg(napl_base):
    r"""
    Delay a spike tensor with an alternating-initialized shift register.

    The operation delays the input stream by **depth** timesteps,

    .. math::

       y_t = x_{t-\mathit{depth}}.

    The first **depth** outputs come from the alternating reset contents, whose
    entry :math:`j` holds :math:`j \bmod 2`.

    An optional per-element ``mask_enable`` selects which elements advance. An
    element whose mask entry is falsy neither stores the current input nor
    advances, so it re-emits the same value on the next call.

    Both buffers take their shape from the first input, and a mid-stream
    ``state_dict`` does not load into a fresh instance.

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
        super().__init__(config, ['depth'], optional_key_list=['polarity'], polarity_required=False)

        #: Number of timesteps retained by the shift register.
        self.depth = config['depth']
        #: Register holding the last :attr:`depth` timesteps, seeded with an alternating pattern.
        self.reg: torch.Tensor
        self.register_buffer('reg', torch.zeros(self.depth, dtype=self.stype))
        for i in range(self.depth):
            self.reg[i].fill_(i%2)
        #: Circular slot each element reads and then replaces, one index per element.
        self.head: torch.Tensor
        self.register_buffer('head', torch.zeros((), dtype=torch.long))
        # reg[head] is depth cycles old; RTL reset uses the same alternating i % 2 pattern.
        #: Hardware latency and timing metadata, with latency equal to :attr:`depth`.
        self.hw = hw_params(pp_delay=self.depth)

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the alternating register contents and the per-element indices.

        Both buffers return to their unexpanded shape and take the next input's
        shape on demand.
        """
        self.reg.resize_(self.depth).zero_()
        for i in range(self.depth):
            self.reg[i].fill_(i%2)
        self.head.resize_(()).zero_()


    def forward(self, input: torch.tensor, mask_enable: torch.tensor=None):
        """
        Push one input timestep through the shift register.

        Args:
            input: Current spike tensor.
            mask_enable: Optional per-element enable with the same shape as ``input``.
                Elements where it is truthy advance; the rest hold their stored
                entry. The default is ``None``, which advances every element.

        Returns:
            The tensor stored ``depth`` timesteps earlier, with the same shape as
            ``input``. The first ``depth`` calls return the alternating reset
            pattern. The current input replaces the entry that was read.

        **Example:**

        .. code-block:: python

            output = delay(torch.tensor([1], dtype=torch.int8))
        """
        # A scalar input makes the expanded shape (depth,), which the reset pattern
        # already has, so the target shape rather than the rank decides.
        if self.reg.shape != (self.depth, *input.shape):
            self.reg.resize_([self.depth, *input.shape]).zero_()
            for i in range(self.depth):
                self.reg[i].fill_(i%2)
            self.head.resize_(input.shape).zero_()

        index = self.head.unsqueeze(0)
        output = self.reg.gather(0, index).squeeze(0)
        if mask_enable is None:
            self.reg.scatter_(0, index, input.detach().unsqueeze(0))
            self.head.add_(1).remainder_(self.depth)
        else:
            if mask_enable.shape != input.shape:
                message = (
                    f'Enable mask shape <{tuple(mask_enable.shape)}> does not match the '
                    f'input shape <{tuple(input.shape)}>.')
                logger.error(message)
                raise AssertionError(message)
            enabled = mask_enable.bool()
            # Writing the value just read back into a held slot leaves it unchanged,
            # so one scatter covers both cases and the indices stay tensor-side.
            stored = torch.where(enabled, input.detach(), output)
            self.reg.scatter_(0, index, stored.unsqueeze(0))
            self.head.copy_(torch.where(enabled, (self.head + 1) % self.depth, self.head))
        return output

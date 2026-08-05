import torch
from loguru import logger

from napl.sim.base import napl_base, hw_params


class shiftreg(napl_base):
    r"""
    Delay a spike tensor with an alternating-initialized shift register.

    Let R_t be the depth-row register and h_t a per-element circular index.
    With R_0[j] = j mod 2 and h_0 = 0, each element e reads its own slot before
    any update, so the exact recurrence is

    .. math::

       \begin{aligned}
       y_t[e] &= R_t[h_t[e], e],\\
       R_{t+1}[h_t[e], e] &= x_t[e],\qquad
       h_{t+1}[e] = (h_t[e]+1) \bmod depth.
       \end{aligned}

    After the alternating contents are emitted, each call returns the input
    from depth timesteps earlier.

    An optional per-element ``mask_enable`` m_t selects which elements advance,
    so the recurrence applies only where it is truthy,

    .. math::

       (R_{t+1}, h_{t+1})[e] = \begin{cases}
       (x_t[e],\; (h_t[e]+1) \bmod depth), & m_t[e] \ne 0,\\
       (R_t[h_t[e], e],\; h_t[e]), & \text{otherwise},
       \end{cases}

    so a held element neither stores nor advances and re-emits the same oldest
    value on the next call.

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
        super().__init__(config, ['depth'], polarity_required=False)

        #: Number of timesteps retained by the shift register.
        self.depth = config['depth']
        #: Alternating reset pattern and device anchor for the register.
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
                column. The default is ``None``, which advances every element.

            Returns:
            The oldest stored tensor. Initial calls return the alternating reset
            pattern; later calls return prior inputs. The current input is copied
            into the register.

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
            assert mask_enable.shape == input.shape, logger.error(
                f'Enable mask shape <{tuple(mask_enable.shape)}> does not match the '
                f'input shape <{tuple(input.shape)}>.')
            enabled = mask_enable.bool()
            # Writing the value just read back into a held slot leaves it unchanged,
            # so one scatter covers both cases and the indices stay tensor-side.
            stored = torch.where(enabled, input.detach(), output)
            self.reg.scatter_(0, index, stored.unsqueeze(0))
            self.head.copy_(torch.where(enabled, (self.head + 1) % self.depth, self.head))
        return output

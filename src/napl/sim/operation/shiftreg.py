import torch

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

        #: Number of timesteps retained by the shift register.
        self.depth = config['depth']
        # reg[head] is depth cycles old; RTL reset uses the same alternating i % 2 pattern.
        #: Hardware latency and timing metadata, with latency equal to :attr:`depth`.
        self.hw = hw_params(pp_delay=self.depth)
        #: Alternating reset pattern and device anchor for the register.
        self.reg: torch.Tensor
        self.register_buffer('reg', torch.zeros(self.depth, dtype=self.stype))
        for i in range(self.depth):
            self.reg[i].fill_(i%2)
        #: Circular index of the register row replaced on the next call.
        self.register_buffer('head', torch.zeros((), dtype=torch.long))
        #: Host-side circular index used for per-timestep tensor indexing.
        self._head = 0
        #: Whether the register must be expanded for the next input shape.
        self.is_first_call = True


    def _reset(self):
        """
        Restore alternating register contents and restart the circular index.
        """
        self.reg.resize_(self.depth).zero_()
        for i in range(self.depth):
            self.reg[i].fill_(i%2)
        self.head.zero_()
        self._head = 0
        self.is_first_call = True


    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
        """Resize the dynamic register before loading a saved stream state."""
        reg_key = prefix + 'reg'
        if reg_key in state_dict:
            self.reg.resize_as_(state_dict[reg_key])
            self.is_first_call = state_dict[reg_key].ndim == 1
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._head = int(self.head.item())


    def forward(self, input: torch.tensor):
        """
        Push one input timestep through the shift register.

        Args:
            input: Current spike tensor.

            Returns:
            The oldest stored tensor. Initial calls return the alternating reset
            pattern; later calls return prior inputs. The current input is copied
            into the register.

        **Example:**

        .. code-block:: python

            output = delay(torch.tensor([1], dtype=torch.int8))
        """
        if self.is_first_call:
            input_shape = list(input.shape)
            input_shape.insert(0, self.depth)
            self.reg.resize_(input_shape).zero_()
            for i in range(self.depth):
                self.reg[i].fill_(i%2)
            self._head = int(self.head.item())
            self.is_first_call = False

        head = self._head
        output = self.reg[head].clone()
        self.reg[head].copy_(input.detach())
        self._head = (head + 1) % self.depth
        self.head.fill_(self._head)
        return output

import torch

from napl.sim.base import napl_base, hw_params


class dff(napl_base):
    r"""
    Delay a spike tensor by a fixed number of timesteps.

    Let R_t be the depth-row FIFO and h_t its circular head. With
    R_0[j] = 0 and h_0 = 0, the exact delay recurrence is

    .. math::

       \begin{aligned}
       y_t &= R_t[h_t],\\
       R_{t+1}[h_t] &= x_t,\qquad
       h_{t+1}=(h_t+1)\bmod depth.
       \end{aligned}

    Thus the first depth outputs are zero and later outputs are the
    corresponding earlier inputs.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import dff

        delay = dff({'depth': 1})
        first = delay(torch.tensor([1], dtype=torch.int8))
        second = delay(torch.tensor([0], dtype=torch.int8))
    """


    def __init__(
            self,
            config={'depth': 1}
        ):
        """
        Configure the delay length.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **depth**: Number of timesteps to delay the input; the default is ``1``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['depth'], polarity_required=False)
        #: Number of timesteps between an input and its delayed output.
        self.depth = config['depth']
        # This buffer anchors lazy state to the module device.
        #: Device and spike-dtype anchor used when the delay queue is initialized.
        self.reg: torch.Tensor
        self.register_buffer('reg', torch.zeros(1, dtype=self.stype))
        # FIFO rows hold tensor references and are overwritten through a circular index.
        #: Circular list of delayed tensor snapshots, allocated on first use.
        self.buf = None
        #: Whether the delay queue must be allocated for the next input shape.
        self.is_first_call = True
        #: Circular index of the oldest delayed tensor to emit and replace.
        self.head = 0
        #: Hardware latency and timing metadata, with latency equal to :attr:`depth`.
        self.hw = hw_params(pp_delay=self.depth)

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Discard the local delay queue and restore its initial position.
        """
        self.buf = None
        self.is_first_call = True
        self.head = 0


    def forward(self, input: torch.tensor):
        """
        Push one input timestep through the delay line.

        Args:
            input: Spike tensor for the current timestep.

        Returns:
            The tensor supplied ``depth`` calls earlier, or zeros while the
            delay line is filling. The input is copied into local queue state.

        **Example:**

        .. code-block:: python

            delayed = delay(torch.tensor([1], dtype=torch.int8))
        """
        if self.is_first_call:
            zero = torch.zeros_like(input, device=self.reg.device)
            self.buf = [zero.clone() for _ in range(self.depth)]
            self.is_first_call = False

        # Emit the oldest row before replacing it with the current input snapshot.
        output = self.buf[self.head]
        self.buf[self.head] = input.detach().clone()
        self.head += 1
        if self.head == self.depth:
            self.head = 0
        return output

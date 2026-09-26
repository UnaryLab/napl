import torch
from loguru import logger

from napl.sim.base import napl_base


class delay(napl_base):
    r"""
    Delay a spike tensor by a fixed number of timesteps.

    The operation delays the input stream by **depth** timesteps,

    .. math::

       y_t = x_{t-\mathit{depth}}.

    The first **depth** outputs come from the initial register contents, which
    **init** selects: ``"zero"`` holds zeros, and ``"alternate"`` holds
    :math:`j \bmod 2` in entry :math:`j`, a rate-0.5 warm-up that is neutral in
    bipolar coding.

    The register takes its shape from the first input, and a mid-stream
    ``state_dict`` does not load into a fresh instance.

    The stored and returned tensors carry the spike type, so a float input
    returns a spike-type tensor rather than a float one.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import delay

        line = delay({'depth': 2})
        output = line(torch.tensor([1], dtype=torch.int8))
    """
    #: Dominant hardware mechanism of this class.
    mechanism = 'delay'


    def __init__(
            self,
            config={'depth': 1}
        ):
        """
        Configure the delay length and its initial register contents.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **depth**: Number of stored timesteps and output delay cycles; the default is ``1``.
              - **init**: Initial register contents, either ``"zero"`` or ``"alternate"``; the default is ``"zero"``.
              - **polarity**: Stream encoding, either ``"unipolar"`` or ``"bipolar"``; optional and unused by the delay itself.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['depth'], optional_key_list=['init', 'polarity'],
                         polarity_required=False)

        #: Number of timesteps retained by the delay line.
        self.depth = config['depth']
        if type(self.depth) is not int or self.depth < 1:
            message = f'Invalid depth: <{self.depth}>; legal values: an integer of at least 1.'
            logger.error(message)
            raise AssertionError(message)
        #: Initial register contents, either ``"zero"`` or ``"alternate"``.
        self.init = config.get('init', 'zero')
        if self.init not in ('zero', 'alternate'):
            message = f'Invalid init: <{self.init}>; legal values: ["zero", "alternate"].'
            logger.error(message)
            raise AssertionError(message)

        #: Register holding the last :attr:`depth` timesteps.
        self.reg: torch.Tensor
        self.register_buffer('reg', torch.zeros(self.depth, dtype=self.stype))
        self._fill()
        #: Circular slot each element reads and then replaces, one index per element.
        self.head: torch.Tensor
        self.register_buffer('head', torch.zeros((), dtype=torch.long))
        # reg[head] is depth cycles old, so the modeled latency equals depth.
        #: Hardware latency and timing metadata, with latency equal to :attr:`depth`.
        self.hw.pp_delay = self.depth

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}


    def _reset(self):
        """
        Restore the initial register contents and the per-element indices.

        Both buffers return to their unexpanded shape and take the next input's
        shape on demand.
        """
        self.reg.resize_(self.depth)
        self._fill()
        self.head.resize_(()).zero_()


    def forward(self, input: torch.Tensor):
        """
        Push one input timestep through the delay line.

        Args:
            input: Current spike tensor.

        Returns:
            The tensor stored ``depth`` timesteps earlier, with the same shape as
            ``input``. The first ``depth`` calls return the configured initial
            pattern. The current input replaces the entry that was read.

        **Example:**

        .. code-block:: python

            output = line(torch.tensor([1], dtype=torch.int8))
        """
        # A scalar input makes the expanded shape (depth,), which the initial pattern
        # already has, so the target shape rather than the rank decides.
        if self.reg.shape != (self.depth, *input.shape):
            self.reg.resize_([self.depth, *input.shape])
            self._fill()
            self.head.resize_(input.shape).zero_()

        input_stype = input.type(self.stype)
        index = self.head.unsqueeze(0)
        output = self.reg.gather(0, index).squeeze(0)
        self.reg.scatter_(0, index, input_stype.detach().unsqueeze(0))
        self.head.add_(1).remainder_(self.depth)
        return output


    def _fill(self):
        """Write the configured initial pattern into the register."""
        self.reg.zero_()
        if self.init == 'alternate':
            for i in range(self.depth):
                self.reg[i].fill_(i%2)

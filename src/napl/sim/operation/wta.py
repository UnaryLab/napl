import torch

from napl.sim.base import napl_base


class wta(napl_base):
    r"""
    Emit one spike at the earliest falling edge across stacked temporal inputs.

    NAPL temporal streams start high and fall once; the falling edge carries the
    encoded value. This operation accepts one tensor with the input streams
    stacked along **dim** (``-1`` by default), emits a single output spike when
    the first input falling edge arrives, and suppresses every later edge until
    :meth:`reset` is called. Simultaneous earliest edges are merged into one
    output spike. If no input falls, the output remains zero.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import wta

        operation = wta()
        input_t = torch.tensor([[1, 1], [1, 0]], dtype=torch.int8)
        output = operation(input_t)

    .. container:: api-references

        .. rubric:: References

        *Race Logic: A hardware acceleration for dynamic programming algorithms*, ISCA, 2014.

        *Space-Time Computing with Temporal Neural Networks*, Synthesis Lectures on Computer Architecture, 2017.
    """


    def __init__(
            self,
            config = {}
    ):
        """
        Construct the streaming temporal winner-take-all operation.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping with no operation-specific keys.
              **polarity** and **name** are optional; the default is ``{}``.
        """
        super().__init__(config, [], optional_key_list=['polarity'], polarity_required=False)

        #: Previous timestep for every stacked input, initialized high so a zero
        #: first timestep is recognized as an earliest falling edge.
        self.previous: torch.Tensor
        self.register_buffer('previous', torch.ones(1, dtype=self.stype))
        #: Per-element latch that suppresses output after the first winner.
        self.fired: torch.Tensor
        self.register_buffer('fired', torch.zeros(1, dtype=self.stype))
        #: Stacked input shape established by the first call in a run.
        self.input_shape = None
        #: Normalized reduction dimension established by the first call.
        self.input_dim = None
        #: Whether the input shape and reduction dimension still need initialization.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the winner-take-all output.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'tc', 'output': 'tc'}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the winner latch and temporal history to their initial state.
        """
        self.previous.resize_(1).fill_(1)
        self.fired.resize_(1).zero_()
        self.input_shape = None
        self.input_dim = None
        self.is_first_call = True


    def forward(self, input, dim=-1):
        """
        Process one timestep from stacked temporal input streams.

        Args:
            input: A 0/1 tensor containing the current timestep of all temporal
                streams. Streams are stacked along **dim**.
            dim: Dimension containing the stacked streams. Defaults to ``-1``.

        Returns:
            A spike tensor with the stacked dimension removed. It contains one
            output spike at the first input falling edge, or all zeros after
            that edge or when no input falls. The call updates the previous
            input history and winner latch.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([[1, 1], [1, 0]], dtype=torch.int8))
        """
        current = input.to(dtype=self.stype)
        if current.ndim == 0:
            raise ValueError('wta input must include a stacked stream dimension.')
        if dim < -current.ndim or dim >= current.ndim:
            raise IndexError(f'wta dimension {dim} is out of range for input with {current.ndim} dimensions.')
        normalized_dim = dim % current.ndim

        if self.is_first_call:
            self.previous = torch.ones_like(current)
            output_shape = current.shape[:normalized_dim] + current.shape[normalized_dim + 1:]
            self.fired = torch.zeros(output_shape, dtype=self.stype, device=current.device)
            self.input_shape = tuple(current.shape)
            self.input_dim = normalized_dim
            self.is_first_call = False
        else:
            if tuple(current.shape) != self.input_shape:
                raise ValueError('wta input shape cannot change before reset.')
            if normalized_dim != self.input_dim:
                raise ValueError('wta reduction dimension cannot change before reset.')

        falling = self.previous.gt(0) & current.eq(0)
        candidate = falling.any(dim=self.input_dim).to(self.stype)
        output = torch.where(self.fired.eq(0), candidate, torch.zeros_like(candidate))

        self.fired.bitwise_or_(output)
        self.previous.copy_(current)
        return output

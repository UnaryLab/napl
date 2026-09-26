import torch

from loguru import logger

from napl.sim.base import napl_base


class argmax(napl_base):
    r"""
    Select the largest running rate from stacked rate-coded streams.

    The target rate-domain operation is

    .. math::

       \mathbf{y} = \operatorname{one\_hot}\!\left(
       \operatorname*{arg\,max}_{k} p_k\right).

    Streams are stacked along **dim**. Each call emits a same-shaped unipolar
    one-hot tensor from the running spike counts held before that timestep,
    then adds the current spikes to bounded unsigned counters. Ties select the
    lowest lane index.

    The counters exactly represent cumulative spike counts until a lane
    reaches ``2 ** width - 1``. A saturated lane stays at that value, so later
    inputs cannot restore information lost to saturation. Choose **width** for
    the required unsaturated stream horizon. Use this operation when a
    downstream stage needs K-way maximum winner streams.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import argmax

        select = argmax({'dim': -1, 'width': 32})
        winner = select(torch.tensor([[1, 0, 1]], dtype=torch.int8))
    """
    #: Dominant hardware mechanism of this class.
    mechanism = 'finite-state-machine'


    def __init__(
            self,
            config={
                'dim': -1,
                'width': 32,
            }
    ):
        """
        Configure the lane dimension and counter width.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Optional input rate-encoding metadata,
                ``"unipolar"`` or ``"bipolar"``. Selection depends only on
                spike counts, so omitting it does not change the result.
              - **dim**: Dimension containing the candidate streams; the
                default is ``-1``.
              - **width**: Unsigned counter width from 1 to 63 bits; the
                default is ``32``.
              - **name**: Optional instance label.
        """
        super().__init__(
            config, [], optional_key_list=['polarity', 'dim', 'width'],
            polarity_required=False,
        )

        #: Configured dimension containing the candidate streams.
        self.dim = config.get('dim', -1)
        if type(self.dim) is not int:
            message = f'Invalid dim: <{self.dim}>; legal values: an integer.'
            logger.error(message)
            raise AssertionError(message)

        #: Width of each unsigned saturating spike counter.
        self.width = config.get('width', 32)
        if type(self.width) is not int or not 1 <= self.width <= 63:
            message = (
                f'Invalid width: <{self.width}>; legal values: an integer '
                'from 1 to 63.'
            )
            logger.error(message)
            raise AssertionError(message)
        #: Largest count retained by the unsigned saturating counters.
        self.count_max = 2**self.width - 1

        #: Running spike count for every input lane.
        self.count: torch.Tensor
        self.register_buffer('count', torch.zeros(1, dtype=torch.long))
        #: Input shape established by the first call in a run.
        self.input_shape = None
        #: Normalized candidate-stream dimension established by the first call.
        self.input_dim = None
        #: Whether input-dependent state must be initialized on the next call.
        self.is_first_call = True

        #: Hardware latency and timing metadata for the registered decision.
        self.hw.pp_delay = 1

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'output': 'unipolar'}
        self.correlation_i = {}


    def _reset(self):
        """
        Clear the running counts and input-shape state.
        """
        self.count.resize_(1).zero_()
        self.input_shape = None
        self.input_dim = None
        self.is_first_call = True


    def forward(self, input):
        """
        Select one winner from a timestep of stacked rate-coded streams.

        Args:
            input: Current 0/1 spike tensor. Candidate streams occupy **dim**;
                the tensor must have rank at least one and at least two lanes.

        Returns:
            A same-shaped unipolar tensor containing one winner spike along
            **dim**. The winner reflects the bounded counts held before this
            timestep; the call then adds the current spikes with saturation.

        **Example:**

        .. code-block:: python

            winner = select(torch.tensor([[1, 0, 1]], dtype=torch.int8))
        """
        name = type(self).__name__
        if input.ndim < 1:
            raise ValueError(f'{name} input must have rank at least 1.')
        if self.dim < -input.ndim or self.dim >= input.ndim:
            raise IndexError(
                f'{name} dimension {self.dim} is out of range for input '
                f'with {input.ndim} dimensions.'
            )
        normalized_dim = self.dim % input.ndim
        if input.size(normalized_dim) < 2:
            raise ValueError(f'{name} requires at least 2 candidate streams.')

        if self.is_first_call:
            self.count.resize_as_(input).zero_()
            self.input_shape = tuple(input.shape)
            self.input_dim = normalized_dim
            self.is_first_call = False
        elif tuple(input.shape) != self.input_shape:
            raise ValueError(f'{name} input shape cannot change before reset.')

        # Torch extrema resolve equal counts to the lowest lane index.
        winner = torch.argmax(self.count, dim=self.input_dim, keepdim=True)
        output = torch.zeros_like(input, dtype=self.stype)
        output.scatter_(self.input_dim, winner, 1)
        self.count.add_(input.to(torch.long)).clamp_(max=self.count_max)
        return output

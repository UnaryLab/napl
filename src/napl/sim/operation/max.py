import torch

from napl.sim.base import napl_base
from .sync_skewed import sync_skewed
from loguru import logger


class max(napl_base):
    r"""
    Select the maximum of two rate-coded streams and track its source.

    The target rate-domain operation is

    .. math::

       y = \max(p_0,p_1).

    Alongside the selected stream, the call returns the running argmax index,
    where ``0`` denotes ``input_0`` and ``1`` denotes ``input_1``. The
    selection state starts at ``0``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import max

        maximum = max()
        spike, index = maximum(torch.tensor([1], dtype=torch.int8),
                               torch.tensor([0], dtype=torch.int8))
    """
    #: Dominant hardware mechanism of this class.
    mechanism = 'finite-state-machine'


    def __init__(
            self,
            config = {}
    ):
        """
        Construct the selector with its fixed-width synchronizer.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping with no operation-specific keys.
              **name** may optionally label the instance; the default is ``{}``.
        """
        super().__init__(config, [], optional_key_list=['polarity'], polarity_required=False)

        #: Previous selection decision used to route the maximum stream to the output.
        self.index: torch.Tensor
        self.register_buffer('index', torch.zeros(1, dtype=torch.int8))
        #: Skew synchronizer that correlates the two input streams before selection.
        self.sync = sync_skewed({'width': 2})
        #: Hardware latency and timing metadata for the combinational maximum output.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc', 'index': 'rc'}
        self.polarity_io = {}
        self.correlation_i = {}


    def _reset(self):
        """
        Restore the local selection state to choose the first input.
        """
        self.index.resize_(1).zero_()


    def forward(self, input_0, input_1):
        """
        Select one output spike and update the running argmax.

        Args:
            input_0: Current 0/1 spikes from the first stream.
            input_1: Current 0/1 spikes from the second stream.

        Returns:
            A pair ``(output, index)``. ``output`` is selected using the prior
            state, while ``index`` is the updated argmax, where ``0`` denotes the
            first input and ``1`` denotes the second.

        **Example:**

        .. code-block:: python

            spike, index = maximum(torch.tensor([1], dtype=torch.int8),
                                   torch.tensor([0], dtype=torch.int8))
        """
        sync_0, sync_1 = self.sync(input_0, input_1)
        sync_0_i8 = sync_0.type(torch.int8)
        sync_1_i8 = sync_1.type(torch.int8)
        d_enable = sync_0_i8 ^ sync_1_i8

        # The selection state is registered at rank 1 and takes the timestep shape on the
        # first call after construction or reset, so the routing below sees the input
        # shape. A 0-dim timestep on that first call leaves no dimension for the rank-1 state
        # to grow into and is rejected below. A mid-run shape change is not rejected here, and
        # the rule it has to satisfy is set by self.sync, which holds its skew counter in place:
        # the new shape must broadcast into the counter shape, not merely broadcast with it.
        # A shape that does not broadcast with the counter at all fails at sync_skewed.py:127
        # (`select = cnt_not_min - ...`), and a shape that does broadcast with the counter
        # but is wider than it fails at sync_skewed.py:130 (`self.cnt.add_(diff)`, an
        # in-place update whose output keeps the counter shape). Both surface as a torch
        # RuntimeError from sync_skewed rather than an AssertionError raised here. A shape
        # that broadcasts into the counter, such as a shape-(1,) operand against a shape-(3,)
        # counter, reaches this line with d_enable already at the state shape, so the stale
        # selection state is kept.
        if self.index.shape != d_enable.shape:
            if d_enable.dim() == 0:
                message = (
                    'Invalid max operand rank: <0>; legal values: at least 1, since the '
                    'selection state is held at rank 1.'
                )
                logger.error(message)
                raise AssertionError(message)
            grown = self.index.expand_as(d_enable).clone()
            self.index.resize_as_(grown).copy_(grown)

        # Output uses the prior selection state; the returned index uses the updated state.
        output = input_0 + self.index * (input_1 - input_0)

        self.index.add_(d_enable * (sync_1_i8 - self.index))

        return output.type(self.stype), self.index.type(self.stype)

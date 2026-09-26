import torch

from napl.sim.base import napl_base
from .sync_skewed import sync_skewed
from loguru import logger


class gt(napl_base):
    r"""
    Compare two rate-coded streams for a greater-than result.

    The target rate-domain operation is

    .. math::

       y = \mathbf{1}\{p_0 > p_1\}.

    A one output denotes the greater running rate for ``input_0``. The
    decision state starts at ``1`` and each call returns the state held before
    that timestep's update.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import gt

        compare = gt()
        result = compare(torch.tensor([1], dtype=torch.int8),
                         torch.tensor([0], dtype=torch.int8))
    """
    #: Dominant hardware mechanism of this class.
    mechanism = 'finite-state-machine'


    def __init__(
            self,
            config = {}
    ):
        """
        Construct the comparator with its fixed-width synchronizer.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping with no operation-specific keys.
              **name** may optionally label the instance; the default is ``{}``.
        """
        super().__init__(config, [], optional_key_list=['polarity'], polarity_required=False)

        #: Previous comparison decision used as the one-cycle delayed output.
        self.decision: torch.Tensor
        self.register_buffer('decision', torch.ones(1, dtype=torch.int8))
        #: Skew synchronizer that correlates the two input streams before comparison.
        self.sync = sync_skewed({'width': 2})
        #: Hardware latency and timing metadata for the registered comparator output.
        self.hw.pp_delay = 1

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'output': 'unipolar'}
        self.correlation_i = {}


    def _reset(self):
        """
        Restore the local greater-than decision state to ``1``.
        """
        self.decision.resize_(1).fill_(1)


    def forward(self, input_0, input_1):
        """
        Compare one timestep from two rate-coded streams.

        Args:
            input_0: Current 0/1 spikes from the first stream.
            input_1: Current 0/1 spikes from the second stream.

        Returns:
            The decision state from before this timestep, where ``1`` denotes
            ``input_0 > input_1`` by running rate. The call updates that state.

        **Example:**

        .. code-block:: python

            result = compare(torch.tensor([1], dtype=torch.int8),
                             torch.tensor([0], dtype=torch.int8))
        """
        sync_0, sync_1 = self.sync(input_0, input_1)
        sync_0_i8 = sync_0.type(torch.int8)
        sync_1_i8 = sync_1.type(torch.int8)
        d_enable = sync_0_i8 ^ sync_1_i8

        # The decision state is registered at rank 1 and takes the timestep shape on the
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
        # decision state is kept.
        if self.decision.shape != d_enable.shape:
            if d_enable.dim() == 0:
                message = (
                    'Invalid gt operand rank: <0>; legal values: at least 1, since the '
                    'decision state is held at rank 1.'
                )
                logger.error(message)
                raise AssertionError(message)
            grown = self.decision.expand_as(d_enable).clone()
            self.decision.resize_as_(grown).copy_(grown)

        # Output reflects the pre-update decision state.
        output = self.decision.clone()

        self.decision.add_(d_enable * (sync_0_i8 - self.decision))

        return output.type(self.stype)

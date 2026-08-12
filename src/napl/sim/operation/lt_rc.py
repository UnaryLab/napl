import torch

from napl.sim.base import napl_base
from .sync_skewed import sync_skewed


class lt_rc(napl_base):
    r"""
    Compare two rate-coded streams for a less-than result.

    The target rate-domain operation is

    .. math::

       y = \mathbf{1}\{p_0 < p_1\}.

    A one output denotes the smaller running rate for ``input_0``. The
    decision state starts at ``0`` and each call returns the state held before
    that timestep's update.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import lt_rc

        compare = lt_rc()
        result = compare(torch.tensor([0], dtype=torch.int8),
                         torch.tensor([1], dtype=torch.int8))
    """


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
        self.register_buffer('decision', torch.zeros(1, dtype=torch.int8))
        #: Skew synchronizer that correlates the two input streams before comparison.
        self.sync = sync_skewed({'width': 2})
        #: Hardware latency and timing metadata for the registered comparator output.
        self.hw.pp_delay = 1

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'output': 'unipolar'}
        self.correlation_i = {}


    def _reset(self):
        """
        Restore the local less-than decision state to ``0``.
        """
        self.decision.resize_(1).zero_()


    def forward(self, input_0, input_1):
        """
        Compare one timestep from two rate-coded streams.

        Args:
            input_0: Current 0/1 spikes from the first stream.
            input_1: Current 0/1 spikes from the second stream.

        Returns:
            The decision state from before this timestep, where ``1`` denotes
            ``input_0 < input_1`` by running rate. The call updates that state.

        **Example:**

        .. code-block:: python

            result = compare(torch.tensor([0], dtype=torch.int8),
                             torch.tensor([1], dtype=torch.int8))
        """
        sync_0, sync_1 = self.sync(input_0, input_1)
        sync_0_i8 = sync_0.type(torch.int8)
        sync_1_i8 = sync_1.type(torch.int8)
        d_enable = sync_0_i8 ^ sync_1_i8

        # Output reflects the pre-update decision state.
        output = self.decision.clone()

        if self.decision.shape == d_enable.shape:
            self.decision.add_(d_enable * (sync_1_i8 - self.decision))
        else:
            updated = self.decision + d_enable * (sync_1_i8 - self.decision)
            self.decision.resize_as_(updated).copy_(updated.detach())

        return output.type(self.stype)

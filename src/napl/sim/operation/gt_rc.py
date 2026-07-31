import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import sync_skewed


class gt_rc(napl_base):
    """
    Compare two rate-coded streams for a greater-than result.

    Use this stateful comparator when the relative rates must be inferred from
    streaming spikes. Its skewed synchronizer updates a one-bit decision state,
    and the returned decision has one cycle of modeled latency.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import gt_rc

        compare = gt_rc()
        result = compare(torch.tensor([1], dtype=torch.int8),
                         torch.tensor([0], dtype=torch.int8))
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
        super().__init__(config, [], polarity_required=False)
        self.hw = hw_params(pp_delay=1)

        self.register_buffer('dff', torch.ones(1, dtype=torch.int8))
        # default to optimal width
        self.sync = sync_skewed({'width': 2})


    def _reset(self):
        """
        Restore the local greater-than decision state to ``1``.
        """
        self.dff.resize_(1).fill_(1)


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
        # sync input_0 to input_1
        sync_0, sync_1 = self.sync(input_0, input_1)
        sync_0_i8 = sync_0.type(torch.int8)
        sync_1_i8 = sync_1.type(torch.int8)
        # if sync_0/1 is 01 or 10, enable dff update
        d_enable = sync_0_i8 ^ sync_1_i8

        # generate output
        # if self.dff == 1, input_0 > input_1
        output = self.dff.clone()

        # update the dff if d_enable is 1: mux(d_enable, sync_0, dff)
        # if sync_0/1 is 10, input_0 > input_1, update dff to 1
        if self.dff.shape == d_enable.shape:
            self.dff.add_(d_enable * (sync_0_i8 - self.dff))
        else:
            updated = self.dff + d_enable * (sync_0_i8 - self.dff)
            self.dff.resize_as_(updated).copy_(updated.detach())

        return output.type(self.stype)

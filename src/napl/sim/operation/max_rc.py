import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import sync_skewed


class max_rc(napl_base):
    """
    Select the maximum of two rate-coded streams and track its source.

    Use this streaming maximum when input rates, rather than isolated spikes,
    determine the larger operand. A skewed synchronizer maintains the selection
    state across timesteps.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import max_rc

        maximum = max_rc()
        spike, index = maximum(torch.tensor([1], dtype=torch.int8),
                               torch.tensor([0], dtype=torch.int8))
    """
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
        super().__init__(config, [], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        self.register_buffer('dff', torch.zeros(1, dtype=torch.int8))
        # default to optimal width
        self.sync = sync_skewed({'width': 2})


    def _reset(self):
        """
        Restore the local selection state to choose the first input.
        """
        self.dff.resize_(1).zero_()


    def forward(self, input_0, input_1):
        """
        Select one output spike and update the running argmax.

        Args:
            input_0: Current 0/1 spikes from the first stream.
            input_1: Current 0/1 spikes from the second stream.

        Returns:
            A pair ``(output, index)``. ``output`` is selected using the prior
            state, while ``index`` is the updated state where ``0`` selects the
            first input and ``1`` selects the second.

        **Example:**

        .. code-block:: python

            spike, index = maximum(torch.tensor([1], dtype=torch.int8),
                                   torch.tensor([0], dtype=torch.int8))
        """
        # sync input_0 to input_1
        sync_0, sync_1 = self.sync(input_0, input_1)
        sync_0_i8 = sync_0.type(torch.int8)
        sync_1_i8 = sync_1.type(torch.int8)
        # if sync_0/1 is 01 or 10, enable dff update
        d_enable = sync_0_i8 ^ sync_1_i8

        # generate output before the dff update
        # if self.dff == 1, input_1 is larger, and max is 1
        # mux(dff, input_1, input_0) with fewer elementwise ops (dff is {0,1})
        output = input_0 + self.dff * (input_1 - input_0)

        # update the dff if d_enable is 1: mux(d_enable, sync_1, dff)
        # sync_0/1 is 01, meaning input_0 < input_1
        # this dff value also indicates argmax
        if self.dff.shape == d_enable.shape:
            self.dff.add_(d_enable * (sync_1_i8 - self.dff))
        else:
            updated = self.dff + d_enable * (sync_1_i8 - self.dff)
            self.dff.resize_as_(updated).copy_(updated.detach())

        # if self.dff == 1, input_1 is larger, and max is 1
        return output.type(self.stype), self.dff.type(self.stype)

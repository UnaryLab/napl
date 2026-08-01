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
        #: Hardware latency and timing metadata for the combinational maximum output.
        self.hw = hw_params(pp_delay=0)

        #: Previous selection decision used to route the synchronized maximum stream.
        self.dff: torch.Tensor
        self.register_buffer('dff', torch.zeros(1, dtype=torch.int8))
        #: Skew synchronizer that correlates the two input streams before selection.
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
        sync_0, sync_1 = self.sync(input_0, input_1)
        sync_0_i8 = sync_0.type(torch.int8)
        sync_1_i8 = sync_1.type(torch.int8)
        d_enable = sync_0_i8 ^ sync_1_i8

        # Output uses the prior selection state; the returned index uses the updated state.
        output = input_0 + self.dff * (input_1 - input_0)

        if self.dff.shape == d_enable.shape:
            self.dff.add_(d_enable * (sync_1_i8 - self.dff))
        else:
            updated = self.dff + d_enable * (sync_1_i8 - self.dff)
            self.dff.resize_as_(updated).copy_(updated.detach())

        return output.type(self.stype), self.dff.type(self.stype)

import torch

from napl.sim.base import napl_base, hw_params


class min_tc(napl_base):
    """
    Select the minimum of two temporal-coded streams with an AND gate.

    Use this stateless operation for temporal codes consisting of leading ones
    followed by zeros. The AND of two such streams represents their minimum.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import min_tc

        minimum = min_tc()
        output = minimum(torch.tensor([1], dtype=torch.int8),
                         torch.tensor([0], dtype=torch.int8))
    """
    def __init__(
            self,
            config = {}
    ):
        """
        Construct the stateless temporal-code minimum.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping with no operation-specific keys.
              **name** may optionally label the instance; the default is ``{}``.
        """
        super().__init__(config, [], polarity_required=False)
        #: Hardware latency and timing metadata for the combinational temporal minimum.
        self.hw = hw_params(pp_delay=0)


    def _reset(self):
        """
        Reset no local mutable state.
        """
        pass


    def forward(self, input_0, input_1):
        """
        Compute one timestep of the temporal-code minimum.

        Args:
            input_0: Current 0/1 tensor from the first temporal stream.
            input_1: Current 0/1 tensor from the second temporal stream.

        Returns:
            The elementwise AND tensor. The method changes no local state.

        **Example:**

        .. code-block:: python

            output = minimum(torch.tensor([1], dtype=torch.int8),
                             torch.tensor([0], dtype=torch.int8))
        """
        output = input_0.type(torch.int8) & input_1.type(torch.int8)
        return output.type(self.stype)

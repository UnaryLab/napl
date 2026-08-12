import torch

from napl.sim.base import napl_base


class min_tc(napl_base):
    r"""
    Select the minimum of two temporal-coded streams with an AND gate.

    The target operation is

    .. math::

       y = \min(x_0,x_1).

    napl temporal streams emit ones and then zeros, with the falling edge
    later for larger values, so the elementwise AND keeps the earlier falling
    edge and gives the minimum exactly.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import min_tc

        minimum = min_tc()
        output = minimum(torch.tensor([1], dtype=torch.int8),
                         torch.tensor([0], dtype=torch.int8))

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
        Construct the stateless temporal-code minimum.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping with no operation-specific keys.
              **name** may optionally label the instance; the default is ``{}``.
        """
        super().__init__(config, [], optional_key_list=['polarity'], polarity_required=False)
        #: Hardware latency and timing metadata for the combinational temporal minimum.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'tc', 'input_1': 'tc', 'output': 'tc'}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


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

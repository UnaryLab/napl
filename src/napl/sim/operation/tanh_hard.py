import torch

from napl.sim.base import napl_base


class tanh_hard(napl_base):
    r"""
    Apply hard tanh to an already bounded spike stream.

    The operation is

    .. math::

       y = \operatorname{clip}(x,-1,1).

    Valid unipolar and bipolar spike-stream values already lie within those
    bounds, so the stream passes through unchanged.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import tanh_hard

        operation = tanh_hard()
        output = operation(torch.tensor([0.0, 1.0]))
    """


    def __init__(
        self,
        config={},
    ):
        """
        Construct the stateless streaming identity kernel.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping. It has no class-specific keys; **name** may optionally label the module.
        """
        super().__init__(config, [], optional_key_list=['polarity'], polarity_required=False)
        #: Hardware latency and timing metadata for the combinational hard tanh.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Reset no local state; the kernel has no class-owned mutable state.
        """
        pass


    def forward(self, input: torch.tensor):
        """
        Pass through one timestep of a bounded spike stream.

        The call returns ``input`` unchanged and does not update class-owned state.

        Args:
            input: Tensor of current 0/1 unipolar or bipolar spike bits.

        Returns:
            The same tensor object as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        return input

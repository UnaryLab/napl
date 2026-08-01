import torch

from napl.sim.base import napl_base, hw_params


class tanh_hard(napl_base):
    """
    Apply hard tanh to an already bounded spike stream.

    This streaming kernel is the identity because valid unipolar and bipolar
    spike-stream values already lie within the hard-tanh bounds. Use it as the
    streaming counterpart of a hard-tanh activation without changing spike bits.

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
        super().__init__(config, [], polarity_required=False)
        #: Hardware latency and timing metadata for the combinational hard tanh.
        self.hw = hw_params(pp_delay=0)


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

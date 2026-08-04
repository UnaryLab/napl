import torch

from napl.sim.base import napl_base


class tanh_hub(napl_base):
    r"""
    Apply hard tanh in the binary domain.

    The single-shot forward operation is

    .. math::

       y = \operatorname{clip}(x,-1,1)
       = \min(\max(x,-1),1).

    This is the bounded binary-domain hard-tanh operation.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import tanh_hub

        operation = tanh_hub()
        output = operation(torch.tensor([-2.0, 0.0, 2.0]))
    """
    #: Marks this activation as a single-shot tensor operation.
    streaming = False


    def __init__(
        self,
        config={},
    ):
        """
        Construct the stateless binary-domain hard-tanh kernel.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping. It has no class-specific keys; **name** may optionally label the module.
        """
        super().__init__(config, [])
        #: Modeled scalar latency of the single-shot hard tanh.
        self.delay = 0

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Reset no local state; this single-shot kernel is stateless.
        """
        pass


    def forward(self, input: torch.tensor):
        """
        Clip a complete tensor to ``[-1, 1]``.

        This stateless call does not advance a streaming timestep.

        Args:
            input: Binary-domain tensor to clip.

        Returns:
            Tensor with the same shape as ``input`` and values in ``[-1, 1]``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([-2.0, 0.0, 2.0]))
        """
        return torch.nn.functional.hardtanh(input, -1.0, 1.0)

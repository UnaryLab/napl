import torch

from napl.sim.base import napl_base


class relu_hub(napl_base):
    r"""
    Apply a bounded ReLU in the binary domain.

    The target operation is

    .. math::

       y = \operatorname{clip}(x,0,scale)
       = \min(\max(x,0),scale).

    The call consumes a complete tensor in one shot rather than one spike
    timestep at a time.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import relu_hub

        operation = relu_hub({'scale': 1.0})
        output = operation(torch.tensor([-1.0, 0.5, 2.0]))
    """
    #: Marks this activation as a single-shot tensor operation.
    streaming = False


    def __init__(
            self,
            config={
                'scale': 1.0,
            }
        ):
        """
        Configure the upper clipping bound.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **scale**: Upper output bound; the default is ``1.0``.
              - **name**: Optional module name.
        """
        super().__init__(config, [], optional_key_list=['polarity', 'scale'])
        #: Upper bound applied to the clipped output tensor.
        self.scale = config.get('scale', 1.0)

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Reset no local state; this single-shot kernel is stateless.
        """
        pass


    def forward(self, input):
        """
        Clip a complete tensor to ``[0, scale]``.

        This stateless call does not advance a streaming timestep.

        Args:
            input: Binary-domain tensor to clip.

        Returns:
            Tensor with the same shape as ``input`` and values in ``[0, scale]``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([-1.0, 0.5, 2.0]))
        """
        return torch.nn.functional.hardtanh(input, 0.0, self.scale)

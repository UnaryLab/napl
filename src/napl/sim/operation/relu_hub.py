import torch

from napl.sim.base import napl_base


class relu_hub(napl_base):
    """
    Apply a bounded ReLU in the binary domain.

    This single-shot kernel clips values to ``[0, scale]`` with
    :func:`torch.nn.functional.hardtanh`. Use it for binary-domain training or
    inference when the activation ceiling must be explicit.

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
        super().__init__(config, [])
        #: Upper bound applied to the clipped output tensor.
        self.scale = config.get('scale', 1.0)

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

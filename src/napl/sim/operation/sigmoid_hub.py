import torch

from napl.sim.base import napl_base


class sigmoid_hub(napl_base):
    r"""
    Apply a scaled hard sigmoid in the binary domain.

    The single-shot forward operation is

    .. math::

       y = \operatorname{clip}\left(\frac{scale\,x}{6}+\frac{1}{2},0,1\right).

    This is the piecewise-linear binary-domain sigmoid.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import sigmoid_hub

        operation = sigmoid_hub({'scale': 3})
        output = operation(torch.tensor([-1.0, 0.0, 1.0]))
    """
    #: Marks this activation as a single-shot tensor operation.
    streaming = False


    def __init__(
        self,
        config={
            'scale': 3,
        },
    ):
        """
        Configure the input scale applied before the hard sigmoid.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **scale**: Input multiplier; the default is ``3``.
              - **name**: Optional module name.
        """
        super().__init__(config, [])
        #: Modeled scalar latency of the single-shot hard sigmoid.
        self.delay = 0
        #: Input multiplier applied before the hard sigmoid.
        self.scale = config.get('scale', 3)

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
        Apply the scaled hard sigmoid to a complete tensor.

        This stateless call does not advance a streaming timestep.

        Args:
            input: Binary-domain input tensor.

        Returns:
            Tensor with the same shape as ``input`` and values in ``[0, 1]``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([-1.0, 0.0, 1.0]))
        """
        return torch.nn.functional.hardsigmoid(input * self.scale)

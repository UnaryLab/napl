from napl.sim.base import napl_base, hw_params
from napl.sim.operation import add_any


class relu_sat(napl_base):
    r"""
    Apply ReLU to a bipolar rate-coded stream with saturating adders.

    The precise target rate-domain operation is

    .. math::

       y = \max(x,0).

    With a_0^(1) = a_0^(2) = 0, the two width-three adders implement

    .. math::

       \begin{aligned}
       \tilde a_t^{(1)} &=
       \operatorname{clip}(a_{t-1}^{(1)}+x_t-\tfrac{1}{2},-4,3),&
       u_t &= \mathbf{1}\{\tilde a_t^{(1)}\geq 1\},&
       a_t^{(1)} &= \tilde a_t^{(1)}-u_t,\\
       \tilde a_t^{(2)} &=
       \operatorname{clip}(a_{t-1}^{(2)}+u_t+\tfrac{1}{2},-4,3),&
       y_t &= \mathbf{1}\{\tilde a_t^{(2)}\geq 1\},&
       a_t^{(2)} &= \tilde a_t^{(2)}-y_t.
       \end{aligned}

    The output is the bipolar 0/1 spike y_t produced by the second adder.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import relu_sat

        operation = relu_sat()
        output = operation(torch.tensor([0.0, 1.0]))
    """


    def __init__(
            self,
            config={}
    ):
        """
        Construct the fixed pair of internal saturating adders.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping. It has no class-specific keys; **name** may optionally label the module.
        """
        super().__init__(config, [], polarity_required=False)

        #: Bipolar saturating adder that performs the first ReLU transform stage.
        self.sub_1 = add_any({'polarity': 'bipolar', 'scale': 1, 'width': 3})
        #: Bipolar saturating adder that performs the second ReLU transform stage.
        self.add_1 = add_any({'polarity': 'bipolar', 'scale': 1, 'width': 3})
        #: Hardware latency and timing metadata for the composed ReLU path.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'bipolar', 'output': 'bipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Reset no class-owned state; :meth:`reset` resets the child adders.
        """
        pass


    def forward(self, input):
        """
        Process one timestep of a bipolar rate-coded input stream.

        The call advances both internal saturating adders and returns the ReLU
        output spike.

        Args:
            input: Tensor of current 0/1 input spikes.

        Returns:
            Bipolar 0/1 output spike tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        # With entry=2, the first adder maps input from [-1, 1] to [-1, 0].
        sub_1_out = self.sub_1(input, entry=2, dim=None)
        # The second adder maps sub_1_out from [-1, 0] to [0, 1].
        output = self.add_1(sub_1_out + 1, entry=2, dim=None)
        return output

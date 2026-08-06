from napl.sim.base import napl_base, hw_params
from napl.sim.operation import add_any


class relu_sat(napl_base):
    r"""
    Apply ReLU to a bipolar rate-coded stream with saturating adders.

    The target rate-domain operation is

    .. math::

       y = \max(x,0).

    Rate saturation clips the negative part away, so the result approximates
    the target. The input and the output are both bipolar 0/1 spike streams.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import relu_sat

        operation = relu_sat()
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
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
        super().__init__(config, [], optional_key_list=['polarity'], polarity_required=False)

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

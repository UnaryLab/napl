import torch

from napl.sim.base import napl_base, hw_params


class mul_gaines(napl_base):
    r"""
    Multiply unary streams with the Gaines gate construction.

    Use this stateless operation for two already-decorrelated spike streams:
    AND for unipolar streams and XNOR for bipolar streams.

    For binary input spikes ``s_0`` and ``s_1``, the exact per-timestep gate is

    .. math::

       y_t = \begin{cases}
       s_{0,t} s_{1,t}, & \text{unipolar},\\
       s_{0,t} s_{1,t} + (1-s_{0,t})(1-s_{1,t}), & \text{bipolar}.
       \end{cases}

    Thus the unipolar rate is ``p_0 p_1`` and the bipolar decoded value is the
    product of the two bipolar decoded input values.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mul_gaines

        multiply = mul_gaines({'polarity': 'unipolar'})
        output = multiply(torch.tensor([1], dtype=torch.int8),
                          torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        B. R. Gaines, *Stochastic Computing Systems*, Advances in Information Systems Science, vol. 2, 1969.
    """


    def __init__(
            self,
            config={
                'polarity': 'bipolar',
            }
        ):
        """
        Select the gate from the stream polarity.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Use ``"unipolar"`` for AND or ``"bipolar"`` for XNOR; the default is ``"bipolar"``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity'], polarity_required=True)
        # The unipolar AND and bipolar XNOR paths are combinational.
        #: Hardware latency and timing metadata for the combinational Gaines multiplier.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_0': self.polarity, 'input_1': self.polarity, 'output': self.polarity}
        self.correlation_i = {('input_0', 'input_1'): 'zero'}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Reset no local mutable state.
        """
        pass


    def forward(self, input_0: torch.tensor, input_1: torch.tensor):
        """
        Multiply one timestep of two unary streams.

        Args:
            input_0: Current 0/1 spike tensor from the first stream.
            input_1: Current 0/1 spike tensor from the second stream.

        Returns:
            The elementwise AND or XNOR spike tensor in the configured spike
            dtype. The method changes no local state.

        **Example:**

        .. code-block:: python

            output = multiply(torch.tensor([1], dtype=torch.int8),
                              torch.tensor([1], dtype=torch.int8))
        """
        if self.polarity == 'unipolar':
            return (input_0.type(torch.int8) & input_1.type(torch.int8)).type(self.stype)
        else:
            return input_0.type(torch.int8).bitwise_xor(input_1.type(torch.int8)).bitwise_xor_(1).type(self.stype)

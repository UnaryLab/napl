import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import bi2uni, jkff


class sqrt_tracejkff(napl_base):
    r"""
    Approximate square root with traced bit insertion and a JK flip-flop.

    Use this streaming kernel for unipolar or bipolar rate-coded square root
    when a JK flip-flop trace is desired. Its accuracy is more sensitive to
    input-stream randomness than :class:`sqrt_traceiscb`.

    The precise target rate-domain operation is

    .. math::

       y = \sqrt{x}.

    Let q_t be the stored JK state and x_t the current input spike. The output
    and trace update are

    .. math::

       \begin{aligned}
       u_t &= q_t \mathbin{\lor} x_t, &
       q_{t+1} &= (1-q_t)u_t
       && (\text{unipolar}),\\
       q_{t+1} &= (1-q_t)B(u_t)
       && (\text{bipolar}),\\
       p_q &= \frac{p_u}{p_u+1}, &
       p_u^2 &= p_x,
       \end{aligned}

    where B is the bipolar-to-unipolar conversion used for the bipolar trace.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import sqrt_tracejkff

        operation = sqrt_tracejkff({'polarity': 'unipolar'})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Stochastic Division and Square Root via Correlation*, DAC, 2019.

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*, IEEE Design and Test, 2021.
    """


    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
        },
    ):
        """
        Configure the input encoding and JK-flip-flop trace path.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['polarity'], polarity_required=True)

        #: JK flip-flop that stores the square-root trace state.
        self.jkff = jkff()
        # K is a shape-, device-, and dtype-matched constant-one tensor cached until reset.
        #: Constant-one JK input tensor cached for the current input shape.
        self.jkff_k = None
        if self.polarity == 'bipolar':
            #: Converter that supplies a unipolar magnitude stream in bipolar mode.
            self.bi2uni = bi2uni({'width': 2})
        #: Hardware latency and timing metadata for the composed square-root path.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Clear the cached constant-one input for the JK flip-flop.

        Registered child kernels are reset by :meth:`reset` before this local reset.
        """
        self.jkff_k = None


    def forward(self, input):
        """
        Process one timestep of a rate-coded input stream.

        The call inserts the JK flip-flop trace into the current input and then
        advances the trace path.

        Args:
            input: Tensor of current 0/1 spikes in the configured polarity.

        Returns:
            0/1 square-root output spike tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        trace = self.jkff.q
        # For 0/1 values, ((1 - trace) & input) + trace == trace | input.
        output = (trace | input.type(torch.int8)).type(self.stype)
        if self.polarity == 'unipolar':
            # P_trace = P_out / (P_out + 1).
            self._unipolar_trace(output)
        else:
            # P_trace = (2 * P_out - 1) / ((2 * P_out - 1) + 1).
            out = self.bi2uni(output)
            self._unipolar_trace(out)
        return output


    def _unipolar_trace(self, output):
        """Update the JK flip-flop trace from a unipolar output spike."""
        k = self.jkff_k
        if k is None or k.shape != output.shape or k.device != output.device or k.dtype != output.dtype:
            k = torch.ones_like(output)
            self.jkff_k = k
        self.jkff(output, k)

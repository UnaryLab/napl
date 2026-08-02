import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import bi2uni, jkff


class sqrt_tracejkff(napl_base):
    """
    Approximate square root with traced bit insertion and a JK flip-flop.

    Use this streaming kernel for unipolar or bipolar rate-coded square root
    when a JK flip-flop trace is desired. Its accuracy is more sensitive to
    input-stream randomness than :class:`sqrt_traceiscb`.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import sqrt_tracejkff

        operation = sqrt_tracejkff({'polarity': 'unipolar'})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Stochastic Division and Square Root via Correlation*.

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*.
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
        #: Hardware latency and timing metadata for the composed square-root path.
        self.hw = hw_params(pp_delay=0)

        #: JK flip-flop that stores the square-root trace state.
        self.jkff = jkff()
        # K is a shape-, device-, and dtype-matched constant-one tensor cached until reset.
        #: Constant-one JK input tensor cached for the current input shape.
        self.jkff_k = None
        if self.polarity == 'bipolar':
            #: Converter that supplies a unipolar magnitude stream in bipolar mode.
            self.bi2uni = bi2uni({'width': 2})


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
            self.unipolar_trace(output)
        else:
            # P_trace = (2 * P_out - 1) / ((2 * P_out - 1) + 1).
            out = self.bi2uni(output)
            self.unipolar_trace(out)
        return output


    def unipolar_trace(self, output):
        """
        Update the JK flip-flop trace from a unipolar output spike.

        The method creates or reuses a matching constant-one tensor and advances
        the child JK flip-flop.

        Args:
            output: Tensor of current 0/1 unipolar output spikes.

        Returns:
            ``None``.

        **Example:**

        .. code-block:: python

            operation.unipolar_trace(torch.tensor([0.0, 1.0]))
        """
        k = self.jkff_k
        if k is None or k.shape != output.shape or k.device != output.device or k.dtype != output.dtype:
            k = torch.ones_like(output)
            self.jkff_k = k
        self.jkff(output, k)

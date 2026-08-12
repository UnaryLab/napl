import torch

from napl.sim.base import napl_base
from .bi2uni import bi2uni
from .decorr import decorr
from .jkff import jkff


class sqrt_tracejkff(napl_base):
    r"""
    Approximate square root with traced bit insertion and a JK flip-flop.

    Use this streaming kernel for unipolar or bipolar rate-coded square root
    when a JK flip-flop trace is desired. Its accuracy is more sensitive to
    input-stream randomness than :class:`sqrt_traceiscb`.

    The target rate-domain operation is

    .. math::

       y = \sqrt{x}.

    The kernel approximates it by inserting a trace stream into the input. The
    trace holds the rate :math:`p_q = p_y / (p_y + 1)`, which makes
    :math:`p_y^2 = p_x`, and a JK flip-flop driven by the output stream
    generates it.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import sqrt_tracejkff

        operation = sqrt_tracejkff({'polarity': 'unipolar'})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Stochastic Division and Square Root via Correlation*, DAC, 2019.

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*, IEEE Design & Test, 2021.
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
        # The flip-flop's own output reaches its J input through the inserted trace and collapses the relation, settling the trace at min(u, 0.5) unipolar or p_x / 2 bipolar and returning the input instead of its square root, so this decorrelating shuffle buffer cuts that path to restore an approximate p_q = u / (u + 1); its depth, period, and generator are an empirical local optimum, and the bipolar bound holds only under the tests' Sobol input encoder.
        #: Shuffle buffer that decorrelates the JK input from the stored trace.
        self.decorr = decorr({'polarity': 'unipolar', 'depth': 4,
                              'timestep': 256, 'generator': 'lfsr', 'seed': 1})
        #: Constant-one JK input tensor cached for the current input shape, device, and dtype.
        self.jkff_k = None
        if self.polarity == 'bipolar':
            #: Converter that re-encodes the bipolar output value ``2p - 1`` as a
            #: unipolar rate in bipolar mode.
            self.bi2uni = bi2uni({'width': 2})
        #: Hardware latency and timing metadata for the composed square-root path.
        self.hw.pp_delay = 0
        #: Whether the RTL counterpart must hold its own encoder, true when any part does.
        self.internal_encode = any(part.internal_encode for part in self.children())

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


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
        # decorr returns one reordered stream per port; the second is unused.
        shuffled, _ = self.decorr(output, output)
        k = self.jkff_k
        if k is None or k.shape != shuffled.shape or k.device != shuffled.device or k.dtype != shuffled.dtype:
            k = torch.ones_like(shuffled)
            self.jkff_k = k
        self.jkff(shuffled, k)

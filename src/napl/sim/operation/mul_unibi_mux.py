import torch

from napl.sim.base import napl_base


class mul_unibi_mux(napl_base):
    r"""
    Multiply a unipolar stream by a bipolar stream with a multiplexer.

    Use this operation when one operand is a magnitude in :math:`[0, 1]` carried
    as a unipolar stream and the other is a signed value in :math:`[-1, 1]`
    carried as a bipolar stream. The unipolar stream selects between the bipolar
    stream and a rate-0.5 toggle stream, so the output is bipolar and no
    polarity conversion is needed on either side.

    The target rate-domain operation is

    .. math::

       v_z = p_u v_b = p_u (2 p_b - 1).

    The two input streams must be decorrelated. The toggle stream is a
    deterministic ``0, 1, 0, 1, ...`` sequence clocked on every timestep, so an
    ``input_u`` stream locked to the same parity biases the second multiplexer
    leg away from rate 0.5. This bias is inherent to the design.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import mul_unibi_mux

        multiply = mul_unibi_mux()
        output = multiply(torch.tensor([1], dtype=torch.int8),
                          torch.tensor([1], dtype=torch.int8))
    """


    def __init__(
            self,
            config={}
    ):
        """
        Construct the multiplier and clear its toggle state.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping with no operation-specific keys.
              The polarity of each port is fixed by the operation, so it is not
              configurable. **name** may optionally label the instance; the
              default is ``{}``.
        """
        super().__init__(config, [], optional_key_list=['polarity'], polarity_required=False)

        #: Free-running toggle flip-flop, holding ``0`` after a reset and kept
        #: scalar so it broadcasts against any input shape.
        self.state: torch.Tensor
        self.register_buffer('state', torch.zeros(1, dtype=torch.int8))
        # The output is a gate network over the current inputs and the toggle state.
        #: Hardware latency and timing metadata for the combinational multiplier.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_u': 'rc', 'input_b': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_u': 'unipolar', 'input_b': 'bipolar', 'output': 'bipolar'}
        # The select leg passes input_b through on input_u, so the two need
        # decorrelated operands.
        self.correlation_i = {('input_u', 'input_b'): 'zero'}


    def _reset(self):
        """
        Restore the local toggle flip-flop to the empty state.
        """
        self.state.zero_()


    def forward(self, input_u: torch.Tensor, input_b: torch.Tensor):
        """
        Multiply one timestep of a unipolar and a bipolar stream.

        Args:
            input_u: Current 0/1 spike tensor from the unipolar stream.
            input_b: Current 0/1 spike tensor from the bipolar stream.

        Returns:
            The bipolar product spike tensor in the configured spike dtype. The
            call toggles the flip-flop on every timestep, whatever the inputs
            are.

        **Example:**

        .. code-block:: python

            output = multiply(torch.tensor([1], dtype=torch.int8),
                              torch.tensor([1], dtype=torch.int8))
        """
        in_u = input_u.type(torch.int8)
        in_b = input_b.type(torch.int8)
        # The multiplexer selects input_b at u = 1 and the toggle stream at u = 0,
        # giving p_z = p_u p_b + (1 - p_u) / 2.
        output = torch.where(in_u.bool(), in_b, self.state)
        # A DFF fed by its own inverter emits one on every second timestep.
        self.state ^= 1
        return output.type(self.stype)

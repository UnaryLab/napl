import torch

from napl.sim.base import napl_base


class mul_unibi(napl_base):
    r"""
    Multiply a unipolar stream by a bipolar stream.

    Use this operation when one operand is a magnitude in :math:`[0, 1]` carried
    as a unipolar stream and the other is a signed value in :math:`[-1, 1]`
    carried as a bipolar stream. The output is bipolar, so no polarity
    conversion is needed on either side.

    The target rate-domain operation is

    .. math::

       v_z = p_u v_b = p_u (2 p_b - 1).

    The two input streams must be decorrelated. The output carries a single
    residual, at most :math:`1/(2N)` per element over :math:`N` timesteps and
    always negative, left by an odd number of ``input_u`` zeros.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import mul_unibi

        multiply = mul_unibi()
        output = multiply(torch.tensor([1], dtype=torch.int8),
                          torch.tensor([1], dtype=torch.int8))
    """


    def __init__(
            self,
            config={}
    ):
        """
        Construct the multiplier and clear its divider state.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping with no operation-specific keys.
              The polarity of each port is fixed by the operation, so it is not
              configurable. **name** may optionally label the instance; the
              default is ``{}``.
        """
        super().__init__(config, [], optional_key_list=['polarity'], polarity_required=False)

        #: Per-element divider flip-flop, holding ``0`` after a reset and resized
        #: to the input shape on the first call.
        self.state: torch.Tensor
        self.register_buffer('state', torch.zeros(1, dtype=torch.int8))
        # The output is a gate network over the current inputs and the divider state.
        #: Hardware latency and timing metadata for the combinational multiplier.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_u': 'rc', 'input_b': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_u': 'unipolar', 'input_b': 'bipolar', 'output': 'bipolar'}
        # AND(input_u, input_b) needs decorrelated operands.
        self.correlation_i = {('input_u', 'input_b'): 'zero'}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the local divider flip-flop to the empty state.
        """
        self.state.resize_(1).zero_()


    def forward(self, input_u: torch.Tensor, input_b: torch.Tensor):
        """
        Multiply one timestep of a unipolar and a bipolar stream.

        Args:
            input_u: Current 0/1 spike tensor from the unipolar stream.
            input_b: Current 0/1 spike tensor from the bipolar stream.

        Returns:
            The bipolar product spike tensor in the configured spike dtype. The
            call updates the divider flip-flop on the elements where
            ``input_u`` is low.

        **Example:**

        .. code-block:: python

            output = multiply(torch.tensor([1], dtype=torch.int8),
                              torch.tensor([1], dtype=torch.int8))
        """
        in_u = input_u.type(torch.int8)
        in_b = input_b.type(torch.int8)
        # Each input_u zero is one event clocking the divider.
        event = in_u ^ 1
        # The two disjoint legs give p_z = p_u p_b + (1 - p_u) / 2.
        output = (in_u & in_b) | (event & self.state)
        # Toggling on every event makes the flip-flop emit a one on every second one.
        updated = self.state ^ event
        self.state.resize_as_(updated).copy_(updated.detach())
        return output.type(self.stype)

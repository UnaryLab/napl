import torch

from loguru import logger
from napl.sim.base import hw_params, napl_base
from napl.sim.operation import encode


class relu_tc(napl_base):
    r"""
    Apply ReLU to a bipolar temporal-coded stream.

    The target operation is

    .. math::

       y = \max(x,0).

    Because :math:`\max(x,0)` is the temporal maximum of the input against a
    stream carrying the value zero, the kernel generates that zero reference
    internally with a composed :class:`napl.encode` instance and takes the
    temporal maximum against it, which is the comparison :class:`napl.max_tc`
    performs on two supplied streams.

    napl temporal streams emit ones and then zeros, with the falling edge
    later for larger values, and a bipolar zero falls at the midpoint of the
    ``2**width``-cycle codeword. An input falling later than the midpoint
    therefore passes through unchanged, and an input falling earlier is
    replaced by zero.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import relu_tc

        operation = relu_tc({'width': 8})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """


    def __init__(self, config={'width': 8}):
        """
        Configure the temporal-code width.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Positive temporal-code bit width; the default is ``8``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['width'], optional_key_list=['polarity'], polarity_required=False)

        #: Number of temporal-code bits in one input value.
        self.width = config['width']
        if not isinstance(self.width, int) or self.width <= 0:
            message = f'Invalid width: <{self.width}>; legal values: a positive integer.'
            logger.error(message)
            raise AssertionError(message)
        #: Temporal codeword length in cycles.
        self.len = 2 ** self.width
        #: Reference-stream source; encodes the bipolar zero as a temporal code.
        self.reference_encode = encode({
            'polarity': 'bipolar',
            'timestep': self.len,
            'generator': 'temporal',
        })
        #: Hardware latency and timing metadata for the combinational output path.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'tc', 'output': 'tc'}
        self.polarity_io = {'input': 'bipolar', 'output': 'bipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Reset local state; this kernel holds none.

        The reference stream position lives in the composed encoder, which
        ``reset()`` restarts separately.
        """
        pass


    def forward(self, input: torch.Tensor):
        """
        Process one timestep of a bipolar temporal code.

        The call reads one reference bit from the composed encoder and ORs it
        with ``input``.

        Args:
            input: Tensor of current 0/1 temporal-code bits.

        Returns:
            0/1 output tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        reference_encode_bit = self.reference_encode(input.new_zeros(1))
        output = input.type(torch.int8) | reference_encode_bit.type(torch.int8)
        return output.type(self.stype)

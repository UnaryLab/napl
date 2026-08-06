import torch

from napl.sim.base import napl_base, hw_params


class relu_cnt(napl_base):
    r"""
    Apply ReLU to a bipolar rate-coded stream with a saturating counter.

    The target rate-domain operation is

    .. math::

       y = \max(x,0).

    The output rate is held at or above bipolar zero to the resolution of a
    **width**-bit counter, so the result approximates the target. The input and
    the output are both bipolar 0/1 spike streams.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import relu_cnt

        operation = relu_cnt()
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """


    def __init__(
            self,
            config={
                'width' : 3,
            }
    ):
        """
        Configure the ReLU state counter.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Counter bit width; the default is ``3``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['width'], optional_key_list=['polarity'], polarity_required=False)

        #: Saturating accumulator width in bits.
        self.width = config['width']

        #: Largest value retained by the ReLU accumulator.
        self.buf_max = 2**self.width - 1
        #: Half-scale value that represents bipolar zero.
        self.buf_half = 2**(self.width - 1)
        #: Saturating bipolar accumulator that controls the emitted ReLU spike.
        self.acc: torch.Tensor
        self.register_buffer('acc', torch.zeros(1, dtype=self.ntype).fill_(2**(self.width - 1)))
        #: Hardware latency and timing metadata for the combinational ReLU output.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'bipolar', 'output': 'bipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the accumulator to its half-scale initial state.
        """
        self.acc.resize_(1).fill_(2**(self.width - 1))


    def forward(self, input):
        """
        Process one timestep of a bipolar rate-coded stream.

        The call produces the current ReLU spike, then updates and saturates
        the accumulator from that output spike.

        Args:
            input: Tensor of current 0/1 input spikes.

        Returns:
            Bipolar 0/1 output spike tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        below_half = torch.lt(self.acc, self.buf_half)
        # Bitwise OR promotes the boolean threshold mask to int8.
        output = input.type(torch.int8) | below_half
        # Output uses the accumulator state before this timestep's update.
        if self.acc.shape == output.shape:
            self.acc.add_(output, alpha=2).sub_(1).clamp_(0, self.buf_max)
        else:
            updated = self.acc.add(output, alpha=2).sub_(1).clamp_(0, self.buf_max)
            self.acc.resize_as_(updated).copy_(updated.detach())
        return output.type(self.stype)

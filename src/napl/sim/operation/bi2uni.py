import torch
from loguru import logger

from napl.sim.base import napl_base, hw_params


class bi2uni(napl_base):
    r"""
    Convert a bipolar rate-coded spike stream to unipolar form.

    The output stream carries the same numeric value as the input stream, so its
    rate equals the bipolar value of the input,

    .. math::

       p_y = v_x = 2 p_x - 1.

    A signed accumulator of **width** bits carries the conversion error between
    timesteps. It saturates at its bounds, so the output rate approximates the
    target rather than matching it exactly.

    Only a non-negative input value is representable in unipolar form; a negative
    input value produces an all-zero output stream.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import bi2uni

        converter = bi2uni({'width': 2})
        output = converter(torch.tensor([1, 0], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*, IEEE Design & Test, 2021.
    """


    def __init__(
            self,
            config={
                'width' : 2,
            }
    ):
        """
        Configure the bounded conversion accumulator.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Signed accumulator width in bits; it must make the emission threshold ``1`` reachable, and the default is ``2``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['width'], optional_key_list=['polarity'], polarity_required=False)

        #: Signed conversion-accumulator width in bits.
        self.width = config['width']
        #: Largest value retained by the conversion accumulator.
        self.acc_max = 2**(self.width-1) - 1
        #: Smallest value retained by the conversion accumulator.
        self.acc_min = -2**(self.width-1)
        if self.acc_max < 1:
            message = (
                f'bi2uni width <{self.width}> has accumulator maximum <{self.acc_max}>, '
                'but the emission threshold <1> is unreachable.'
            )
            logger.error(message)
            raise AssertionError(message)
        #: Running bipolar-to-unipolar conversion error.
        self.accumulator: torch.Tensor
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
        #: Hardware latency and timing metadata for the combinational converter.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'bipolar', 'output': 'unipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Clear the local conversion accumulator.
        """
        self.accumulator.resize_(1).zero_()


    def forward(self, input):
        """
        Convert one timestep of bipolar input spikes.

        Args:
            input: Tensor of current 0/1 bipolar-encoded spikes.

        Returns:
            A tensor of unipolar-encoded output spikes with the same shape. The
            call updates the bounded accumulator.

        **Example:**

        .. code-block:: python

            output = converter(torch.tensor([1, 0], dtype=torch.int8))
        """
        # The scalar initial state broadcasts out of place; matching shapes update in place.
        acc = self.accumulator
        if acc.shape == input.shape:
            acc = acc.add_(input, alpha=2).sub_(1).clamp_(self.acc_min, self.acc_max)
        else:
            acc = acc.add(input, alpha=2).sub_(1).clamp_(self.acc_min, self.acc_max)
        output = torch.ge(acc, 1).type(self.stype)
        # Since output is 1 only for acc >= 1, subtraction preserves the accumulator bounds.
        acc.sub_(output)
        if self.accumulator.shape != acc.shape:
            self.accumulator.resize_as_(acc).copy_(acc.detach())
        return output

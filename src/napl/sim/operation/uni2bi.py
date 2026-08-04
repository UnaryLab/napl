import torch
from loguru import logger

from napl.sim.base import napl_base, hw_params


class uni2bi(napl_base):
    r"""
    Convert a unipolar rate-coded spike stream to bipolar form.

    The precise target mapping is

    .. math::

       v_y = 2p_x-1.

    The accumulator starts at ``a_0=0`` with rails ``M_-`` and ``M_+``.
    The exact conversion recurrence is

    .. math::

       \begin{aligned}
       \tilde a_t &= \operatorname{clip}(a_{t-1}+x_t+1,M_-,M_+),\\
       y_t &= \mathbf{1}\{\tilde a_t\geq 2\},\qquad
       a_t=\tilde a_t-2y_t.
       \end{aligned}

    The output y_t is bipolar and preserves the represented value through
    its one-density mapping.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import uni2bi

        converter = uni2bi({'width': 3})
        output = converter(torch.tensor([1, 0], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*, IEEE Design and Test, 2021.
    """


    def __init__(
            self,
            config={
                'width' : 3,
            }
    ):
        """
        Configure the bounded conversion accumulator.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Signed accumulator width in bits; it must make the emission threshold ``2`` reachable, and the default is ``3``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['width'], polarity_required=False)

        #: Signed conversion-accumulator width in bits.
        self.width = config['width']
        #: Largest value retained by the conversion accumulator.
        self.acc_max = 2**(self.width-1) - 1
        #: Smallest value retained by the conversion accumulator.
        self.acc_min = -2**(self.width-1)
        assert self.acc_max >= 2, logger.error(
            f'uni2bi width <{self.width}> has accumulator maximum <{self.acc_max}>, '
            'but the emission threshold <2> is unreachable.'
        )
        #: Running unipolar-to-bipolar conversion error.
        self.accumulator: torch.Tensor
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
        #: Hardware latency and timing metadata for the combinational converter.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'unipolar', 'output': 'bipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Clear the local conversion accumulator.
        """
        self.accumulator.resize_(1).zero_()


    def forward(self, input):
        """
        Convert one timestep of unipolar input spikes.

        Args:
            input: Tensor of current 0/1 unipolar spikes.

        Returns:
            A tensor of bipolar-encoded output spikes with the same shape. The
            call updates the bounded accumulator.

        **Example:**

        .. code-block:: python

            output = converter(torch.tensor([1, 0], dtype=torch.int8))
        """
        # ntype accumulation widens stype input without truncation.
        acc = self.accumulator
        # The scalar initial state broadcasts out of place; matching shapes update in place.
        if acc.shape == input.shape:
            acc.add_(input).add_(1).clamp_(self.acc_min, self.acc_max)
        else:
            acc = acc.add(input).add_(1).clamp_(self.acc_min, self.acc_max)
        output = torch.ge(acc, 2).type(self.stype)
        # Since output is 1 only for acc >= 2, subtracting 2 preserves the bounds.
        acc.sub_(output, alpha=2)
        if self.accumulator.shape != acc.shape:
            self.accumulator.resize_as_(acc).copy_(acc.detach())
        return output

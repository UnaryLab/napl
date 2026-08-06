import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import bi2uni, add_any, shiftreg


class sqrt_emit(napl_base):
    r"""
    Approximate square root by opportunistic bit insertion.

    Use this streaming kernel for unipolar or bipolar rate-coded square root
    when an emission-feedback implementation is desired.

    The target rate-domain operation is

    .. math::

       y = \sqrt{x}.

    The kernel approximates it by inserting extra spikes into the input stream
    whenever the output stream can absorb them, so the accuracy depends on the
    stream length and on the randomness of the input stream.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import sqrt_emit

        operation = sqrt_emit({'polarity': 'unipolar'})
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
        Configure the input encoding and fixed emission-feedback components.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['polarity'], polarity_required=True)

        #: Previous emission bit fed back into the next square-root step.
        self.emit_out: torch.Tensor
        self.register_buffer('emit_out', torch.zeros(1, dtype=self.stype))

        #: Unipolar saturating adder used by the emission update.
        self.nsadd = add_any({'polarity': 'unipolar', 'scale': 1, 'width': 3})
        #: Fixed delay length of the internal emission shift register.
        self.depth = 2
        #: Delay line used by the unipolar emission path.
        self.shiftreg = shiftreg({'depth': self.depth})

        if self.polarity == 'bipolar':
            #: Converter that supplies a unipolar magnitude stream in bipolar mode.
            self.bi2uni = bi2uni({'width': 2})

        #: Whether the next call must expand :attr:`emit_out` to the input shape.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the composed square-root path.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Clear the saved emission bit and restore the first-call state.

        Registered child kernels are reset by :meth:`reset` before this local reset.
        """
        self.emit_out.resize_(1).zero_()
        self.is_first_call = True


    def forward(self, input):
        """
        Process one timestep of a rate-coded input stream.

        The call combines ``input`` with the previous emission bit, advances
        the selected emission path and child kernels, and saves the next
        emission bit.

        Args:
            input: Tensor of current 0/1 spikes in the configured polarity.

        Returns:
            0/1 square-root output spike tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        if self.is_first_call:
            self.emit_out.resize_as_(input).zero_()
            self.is_first_call = False

        # Two 0/1 int8 operands sum exactly; nsadd casts the partial sum to ntype.
        in_sum = input.type(torch.int8) + self.emit_out
        output = self.nsadd(in_sum, dim=None)
        if self.polarity == 'bipolar':
            emit_out = self._bipolar_emit(output)
        else:
            emit_out = self._unipolar_emit(output)
        if self.emit_out.shape == emit_out.shape:
            self.emit_out.copy_(emit_out.detach())
        else:
            self.emit_out.resize_as_(emit_out).copy_(emit_out.detach())

        return output


    def _unipolar_emit(self, output):
        """Generate the next emission bit for a unipolar output spike."""
        output_inv = 1 - output
        output_inv_scrambled = self.shiftreg(output_inv)
        emit_out = output_inv_scrambled.type(torch.int8) & output.type(torch.int8)
        return emit_out


    def _bipolar_emit(self, output):
        """Generate the next emission bit for a bipolar output spike."""
        output_inv = 1 - output
        output_inv_scrambled = self.shiftreg(output_inv)
        output_uni = self.bi2uni(output)
        emit_out = output_inv_scrambled.type(torch.int8) & output_uni.type(torch.int8)
        return emit_out

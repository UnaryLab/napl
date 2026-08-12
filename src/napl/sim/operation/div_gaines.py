import torch

from napl.sim.base import napl_base
from .encode import encode


class div_gaines(napl_base):
    r"""
    Divide rate-coded streams with the Gaines counter construction.

    Use this stateful divider for unipolar or bipolar streams when a stochastic
    quotient is needed directly from an error-integrating saturating counter.
    The output has one cycle of modeled latency.

    The target rate-domain operation is

    .. math::

       y = \frac{x}{d}.

    The dividend and divisor streams must be uncorrelated, and the quotient
    approaches the target as the counter settles, with a residual error set by
    the counter width.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import div_gaines

        divider = div_gaines({'polarity': 'unipolar', 'width': 5,
                              'generator': 'Sobol', 'dim': 1})
        quotient = divider(torch.tensor([1], dtype=torch.int8),
                           torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *Stochastic Computing Systems*, Advances in Information Systems Science, 1969.
    """
    #: The counter-comparison reference is encoded from a held number sequence,
    #: so the RTL counterpart holds its own encoder instead of sharing an
    #: external one.
    internal_encode = True


    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
            'width' : 5,
            'generator' : 'Sobol',
            'dim' : 1,
        }
    ):
        """
        Configure the counter and quotient number sequence.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **width**: Counter width in bits; the default is ``5``.
              - **generator**: Number-sequence generator for quotient thresholds; the default is ``"Sobol"``.
              - **dim**: Generator dimension forwarded when the threshold sequence is built; the default is ``1``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'width', 'generator'], optional_key_list=['dim'], polarity_required=True)

        #: Saturating quotient-counter width in bits.
        self.width = config['width']

        #: Encoder supplying the periodic quotient-spike threshold comparison.
        self.reference_encode = encode({'polarity': 'unipolar',
                                  'timestep': 2 ** self.width,
                                  'generator': config['generator'],
                                  'dim': config.get('dim', 1)})
        scaled = self.reference_encode.num_seq.mul(2 ** self.width)
        assert torch.allclose(scaled, scaled.round(), rtol=0, atol=1e-9), \
            f'Sequence value off the 1/{2 ** self.width} grid; the counter-scale view would not be exact.'
        #: Counter-scale view of the encoder sequence, read by the RTL generator.
        self.rng_seq = scaled.tolist()

        #: Largest value retained by the quotient counter.
        self.scnt_max = 2 ** self.width - 1
        #: Half-scale quotient-counter value restored by ``_reset``.
        self.scnt_init = 2 ** (self.width - 1)
        # The half-scale scalar counter broadcasts to the input shape on first use.
        #: Saturating error counter that controls quotient spike probability.
        self.scnt: torch.Tensor
        self.register_buffer('scnt', torch.full((1,), float(self.scnt_init), dtype=self.ntype))
        # Bipolar feedback uses the delayed divisor to decorrelate the quotient term.
        #: Previous bipolar divisor spike tensor used by the feedback equation.
        self.divisor_d: torch.Tensor
        self.register_buffer('divisor_d', torch.zeros(1, dtype=torch.int8))
        # Output uses the counter state before the current update, giving one-cycle latency.
        #: Hardware latency and timing metadata for the registered divider output.
        self.hw.pp_delay = 1

        self.encoding_io = {'dividend': 'rc', 'divisor': 'rc', 'output': 'rc'}
        self.polarity_io = {'dividend': self.polarity, 'divisor': self.polarity, 'output': self.polarity}
        self.correlation_i = {('dividend', 'divisor'): 'zero'}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the local counter and delayed divisor; :meth:`reset` restarts
        the encoder that holds the sequence position.
        """
        self.scnt.resize_(1).fill_(self.scnt_init)
        self.divisor_d.resize_(1).zero_()


    def forward(self, dividend, divisor):
        """
        Process one dividend and divisor timestep.

        Args:
            dividend: Current 0/1 dividend spike tensor.
            divisor: Current 0/1 divisor spike tensor, broadcast-compatible with
                ``dividend``.

        Returns:
            A quotient spike tensor with the dividend shape. The call advances
            the threshold sequence and updates the counter and delayed divisor.

        **Example:**

        .. code-block:: python

            quotient = divider(torch.tensor([1], dtype=torch.int8),
                               torch.tensor([1], dtype=torch.int8))
        """

        # Dividing by the power-of-two counter scale is exact, so comparing the
        # scaled counter through the encoder reproduces the integer comparison.
        reference_encode_bit = self.reference_encode(self.scnt.div(2 ** self.width))
        output = reference_encode_bit.type(torch.int8)
        if output.shape != dividend.shape:
            output = output.expand(dividend.shape)

        if self.polarity == 'unipolar':
            # Integer spikes promote exactly in the ntype counter update.
            inc = dividend
            dec = output & divisor.type(torch.int8)
        else:
            dividend_i8 = dividend.type(torch.int8)
            divisor_i8 = divisor.type(torch.int8)
            inc = (self.divisor_d ^ divisor_i8 ^ output) ^ 1
            dec = dividend_i8 ^ divisor_i8
            if self.divisor_d.shape == divisor_i8.shape:
                self.divisor_d.copy_(divisor_i8.detach())
            else:
                self.divisor_d.resize_as_(divisor_i8).copy_(divisor_i8.detach())

        # The scalar initial counter broadcasts out of place; matching shapes update in place.
        if self.scnt.shape == inc.shape and self.scnt.shape == dec.shape:
            self.scnt.add_(inc).sub_(dec).clamp_(0, self.scnt_max)
        else:
            updated = self.scnt.add(inc).sub_(dec).clamp_(0, self.scnt_max)
            self.scnt.resize_as_(updated).copy_(updated.detach())

        return output.type(self.stype)

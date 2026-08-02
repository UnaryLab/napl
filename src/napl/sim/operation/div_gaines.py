import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.module import gen_num_seq


class div_gaines(napl_base):
    """
    Divide rate-coded streams with the Gaines counter construction.

    Use this stateful divider for unipolar or bipolar streams when a stochastic
    quotient is needed directly from an error-integrating saturating counter.
    The output has one cycle of modeled latency.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import div_gaines

        divider = div_gaines({'polarity': 'unipolar', 'depth': 5,
                              'generator': 'Sobol', 'dim': 1})
        quotient = divider(torch.tensor([1], dtype=torch.int8),
                           torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        B. R. Gaines, *Stochastic Computing Systems*, 1969.
    """


    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
            'depth' : 5,
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
              - **depth**: Counter width in bits; the default is ``5``.
              - **generator**: Number-sequence generator for quotient thresholds; the default is ``"Sobol"``.
              - **dim**: Generator dimension forwarded when the threshold sequence is built; the default is ``1``.
              - **seed**: Optional LFSR seed used when **generator** is ``"lfsr"``; the default is ``None``.
              - **taps**: Optional LFSR feedback taps used when **generator** is ``"lfsr"``; the default is ``None``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'depth', 'generator'], polarity_required=True)
        # Output uses the counter state before the current update, giving one-cycle latency.
        #: Hardware latency and timing metadata for the registered divider output.
        self.hw = hw_params(pp_delay=1)

        #: Saturating quotient-counter width in bits.
        self.depth = config['depth']
        config['width'] = self.depth

        # Python scalar thresholds span the counter range without device synchronization.
        #: Periodic counter thresholds used to generate quotient spikes.
        self.rng_seq = torch.floor(gen_num_seq(config).mul(2 ** self.depth)).tolist()
        #: Current position in :attr:`rng_seq`.
        self.idx = 0

        #: Largest value retained by the quotient counter.
        self.scnt_max = 2 ** self.depth - 1
        #: Half-scale quotient-counter value restored by :meth:`_reset`.
        self.scnt_init = 2 ** (self.depth - 1)
        # The half-scale scalar counter broadcasts to the input shape on first use.
        #: Saturating error counter that controls quotient spike probability.
        self.scnt: torch.Tensor
        self.register_buffer('scnt', torch.full((1,), float(self.scnt_init), dtype=self.ntype))
        # Bipolar feedback uses the delayed divisor to decorrelate the quotient term.
        #: Previous bipolar divisor spike tensor used by the feedback equation.
        self.divisor_d: torch.Tensor
        self.register_buffer('divisor_d', torch.zeros(1, dtype=torch.int8))


    def _reset(self):
        """
        Restore the local counter, delayed divisor, and sequence position.
        """
        self.idx = 0
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

        output = torch.gt(self.scnt, self.rng_seq[self.idx]).type(torch.int8)
        self.idx = (self.idx + 1) % len(self.rng_seq)
        if output.shape != dividend.shape:
            output = output.expand(dividend.shape)

        if self.polarity == 'unipolar':
            # Integer spikes promote exactly in the ntype counter update.
            inc = dividend
            dec = output & divisor.type(torch.int8)
        else:
            dividend_i8 = dividend.type(torch.int8)
            divisor_i8 = divisor.type(torch.int8)
            # inc - dec = XNOR(divisor_d ^ divisor ^ output) - (dividend ^ divisor).
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

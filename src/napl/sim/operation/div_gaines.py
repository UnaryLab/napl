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
            # depth of the saturating counter; experiments in UnarySim default to 5
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
        # quotient spike is a comparison against the counter register, so one-cycle latency
        self.hw = hw_params(pp_delay=1)

        self.depth = config['depth']
        config['width'] = self.depth

        # rng sequence scaled to integers in [0, 2**depth), matching the counter range
        # static python list of thresholds; indexing with a python scalar per timestep
        # avoids a device-scalar fetch (a per-timestep GPU sync)
        self.rng_seq = torch.floor(gen_num_seq(config).mul(2 ** self.depth)).tolist()
        # index of numbers in the rng seq
        self.idx = 0

        self.scnt_max = 2 ** self.depth - 1
        self.scnt_init = 2 ** (self.depth - 1)
        # saturating up/down counter; scalar that broadcasts to the input shape on the first forward
        self.register_buffer('scnt', torch.full((1,), float(self.scnt_init), dtype=self.ntype))
        # previous divisor spike, used to decorrelate the counter feedback in bipolar mode
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

        # quotient spike from the counter state; identical for both polarities
        output = torch.gt(self.scnt, self.rng_seq[self.idx]).type(torch.int8)
        self.idx = (self.idx + 1) % len(self.rng_seq)
        if output.shape != dividend.shape:
            # first timestep only: counter is still a scalar, expand is a free view
            output = output.expand(dividend.shape)

        if self.polarity == 'unipolar':
            # counter integrates dividend - output * divisor; 0/1 spikes are exact
            # under the int8 -> float promotion of the out-of-place add below
            inc = dividend
            dec = output & divisor.type(torch.int8)
        else:
            dividend_i8 = dividend.type(torch.int8)
            divisor_i8 = divisor.type(torch.int8)
            # XNOR terms: dividend*divisor and output*divisor (via the delayed divisor
            # spike, which decorrelates the feedback from the fresh divisor), folded
            # algebraically into inc - dec = XNOR(divisor_d^divisor^output) - (dividend^divisor)
            inc = (self.divisor_d ^ divisor_i8 ^ output) ^ 1
            dec = dividend_i8 ^ divisor_i8
            if self.divisor_d.shape == divisor_i8.shape:
                self.divisor_d.copy_(divisor_i8.detach())
            else:
                self.divisor_d.resize_as_(divisor_i8).copy_(divisor_i8.detach())

        # saturating up/down counter; the out-of-place add broadcasts up to the input
        # shape on the first call, then sub_/clamp_ mutate that fresh tensor in place
        if self.scnt.shape == inc.shape and self.scnt.shape == dec.shape:
            self.scnt.add_(inc).sub_(dec).clamp_(0, self.scnt_max)
        else:
            updated = self.scnt.add(inc).sub_(dec).clamp_(0, self.scnt_max)
            self.scnt.resize_as_(updated).copy_(updated.detach())

        return output.type(self.stype)

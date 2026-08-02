import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.module import gen_num_seq


class sqrt_gaines(napl_base):
    """
    Approximate square root with the Gaines saturating-counter circuit.

    Use this streaming kernel for unipolar or bipolar rate-coded square root
    when counter state should be sampled by a configurable number sequence.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import sqrt_gaines

        operation = sqrt_gaines({'polarity': 'unipolar', 'width': 5, 'generator': 'Sobol'})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        B. R. Gaines, *Stochastic Computing Systems*.
    """


    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
            'width' : 5,
            'generator' : 'Sobol',
        },
    ):
        """
        Configure the feedback counter and threshold sequence.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **width**: Counter and number-sequence bit width; the default is ``5``.
              - **generator**: Number-sequence generator accepted by :func:`napl.sim.module.encoder.gen_num_seq`; the default is ``"Sobol"``.
              - **dim**: Optional Sobol dimension; the sequence generator defaults to ``1``.
              - **seed**: Optional LFSR seed used when **generator** is ``"lfsr"``; the default is ``None``.
              - **taps**: Optional LFSR tap list used when **generator** is ``"lfsr"``; the default is ``None``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['polarity', 'width', 'generator'], polarity_required=True)
        #: Hardware latency and timing metadata for the registered square-root output.
        self.hw = hw_params(pp_delay=1)

        #: Counter and threshold-sequence width in bits.
        self.width = config['width']
        #: Largest value retained by the square-root counter.
        self.cnt_max = 2**self.width - 1
        #: Half-scale counter value restored by :meth:`_reset`.
        self.cnt_half = 2**(self.width - 1)

        # Python scalar thresholds span the counter range without device synchronization.
        #: Periodic tensor of stochastic counter thresholds.
        self.rand_seq: torch.Tensor
        self.register_buffer('rand_seq', torch.floor(gen_num_seq(config).mul(2**self.width)))
        #: Python-list view of :attr:`rand_seq` used for per-timestep comparison.
        self.rand_seq_vals = self.rand_seq.tolist()
        #: Current position in :attr:`rand_seq_vals`.
        self.idx = 0

        # The half-scale scalar counter broadcasts to the input shape on first use.
        #: Saturating error counter that controls square-root spike generation.
        self.scnt: torch.Tensor
        self.register_buffer('scnt', torch.zeros(1, dtype=self.ntype).fill_(self.cnt_half))
        # Squared-output feedback uses a one-cycle delayed output.
        #: Previous output spike tensor used by the squared-output feedback term.
        self.out_d: torch.Tensor
        self.register_buffer('out_d', torch.zeros(1, dtype=torch.int8))


    def _reset(self):
        """
        Restart the sequence index, counter, and delayed-output feedback state.
        """
        self.idx = 0
        self.scnt.resize_(1).fill_(self.cnt_half)
        self.out_d.resize_(1).zero_()


    def forward(self, input):
        """
        Process one timestep of a rate-coded input stream.

        The call samples the pre-update counter, advances the threshold sequence,
        updates delayed output feedback, and saturates the counter after applying
        the input and squared-output events.

        Args:
            input: Tensor of current 0/1 spikes in the configured polarity.

        Returns:
            0/1 square-root output spike tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        output = torch.gt(self.scnt, self.rand_seq_vals[self.idx]).type(torch.int8)
        self.idx = (self.idx + 1) % len(self.rand_seq_vals)
        if output.shape != input.shape:
            output = output.expand(input.size()).contiguous()

        # Squared-output feedback is AND in unipolar coding and XNOR in bipolar coding.
        if self.polarity == 'unipolar':
            dec = output & self.out_d
        else:
            dec = 1 - (output ^ self.out_d)
        if self.out_d.shape == output.shape:
            self.out_d.copy_(output.detach())
        else:
            self.out_d.resize_as_(output).copy_(output.detach())

        if self.scnt.shape == input.shape and self.scnt.shape == dec.shape:
            self.scnt.add_(input).sub_(dec).clamp_(0, self.cnt_max)
        else:
            updated = (self.scnt + input - dec).clamp(0, self.cnt_max)
            self.scnt.resize_as_(updated).copy_(updated.detach())

        return output.type(self.stype)

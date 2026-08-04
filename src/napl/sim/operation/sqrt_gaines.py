import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import encode


class sqrt_gaines(napl_base):
    r"""
    Approximate square root with the Gaines saturating-counter circuit.

    Use this streaming kernel for unipolar or bipolar rate-coded square root
    when counter state should be sampled by a configurable number sequence.

    The precise target rate-domain operation is

    .. math::

       y = \sqrt{x}.

    Let R_t = round(2**width G_t), where G_t is the configured number
    sequence, and let y_t be sampled before the counter update. With d_{t-1}
    the previous output spike, the exact recurrence is

    .. math::

       \begin{aligned}
       y_t &= \mathbf{1}\{scnt_t > R_t\},\\
       \delta_t &= y_t \mathbin{\land} d_{t-1}
       &&(\text{unipolar}),\\
       \delta_t &= 1-(y_t \oplus d_{t-1})
       &&(\text{bipolar}),\\
       scnt_{t+1} &= \operatorname{clip}(scnt_t+x_t-\delta_t,
       0,2^{width}-1), &
       d_t &= y_t.
       \end{aligned}

    The threshold output uses the pre-update counter and d_t stores that
    output for the next squared-output feedback term.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import sqrt_gaines

        operation = sqrt_gaines({'polarity': 'unipolar', 'width': 5, 'generator': 'Sobol'})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        B. R. Gaines, *Stochastic Computing Systems*, Advances in Information Systems Science, vol. 2, 1969.
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
              - **generator**: Number-sequence generator accepted by :func:`napl.sim.operation.encode.gen_num_seq`; the default is ``"Sobol"``.
              - **dim**: Optional Sobol dimension; the sequence generator defaults to ``1``.
              - **seed**: Optional LFSR seed used when **generator** is ``"lfsr"``; the default is ``None``.
              - **taps**: Optional LFSR tap list used when **generator** is ``"lfsr"``; the default is ``None``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['polarity', 'width', 'generator'], polarity_required=True)

        #: Counter and threshold-sequence width in bits.
        self.width = config['width']
        #: Largest value retained by the square-root counter.
        self.cnt_max = 2**self.width - 1
        #: Half-scale counter value restored by :meth:`_reset`.
        self.cnt_half = 2**(self.width - 1)

        # Python scalar thresholds span the counter range without device synchronization.
        #: Periodic tensor of stochastic counter thresholds.
        #: Encoder supplying the periodic counter-threshold comparison.
        self.reference_encode = encode({'polarity': 'unipolar',
                                   'timestep': 2**self.width,
                                   'generator': config['generator'],
                                   'dim': config.get('dim', 1)})
        scaled = self.reference_encode.num_seq.mul(2**self.width)
        assert torch.allclose(scaled, scaled.round(), atol=1e-9), \
            f'Sequence value off the 1/{2**self.width} grid; the counter-scale view would not be exact.'
        #: Counter-scale view of the encoder sequence, read by the RTL generator.
        self.rand_seq: torch.Tensor
        self.register_buffer('rand_seq', scaled.round())

        # The half-scale scalar counter broadcasts to the input shape on first use.
        #: Saturating error counter that controls square-root spike generation.
        self.scnt: torch.Tensor
        self.register_buffer('scnt', torch.zeros(1, dtype=self.ntype).fill_(self.cnt_half))
        # Squared-output feedback uses a one-cycle delayed output.
        #: Previous output spike tensor used by the squared-output feedback term.
        self.out_d: torch.Tensor
        self.register_buffer('out_d', torch.zeros(1, dtype=torch.int8))
        #: Hardware latency and timing metadata for the registered square-root output.
        self.hw = hw_params(pp_delay=1)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restart the sequence index, counter, and delayed-output feedback state.
        """
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
        # Dividing by the power-of-two counter scale is exact, so comparing the
        # scaled counter through the encoder reproduces the integer comparison.
        reference_encode_bit = self.reference_encode(self.scnt.div(2**self.width))
        output = reference_encode_bit.type(torch.int8)
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

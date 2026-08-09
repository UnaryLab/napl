import torch

from napl.sim.base import napl_base


class sync_skewed(napl_base):
    r"""
    Correlate two unipolar streams with skewed synchronization.

    The first stream is retimed onto the spike positions of the second, and the
    second passes through unchanged. Every spike of the retimed stream then
    coincides with a spike of the second stream, which maximizes the
    correlation of the returned pair.

    The retiming buffers up to :math:`2^{width}-1` unmatched spikes. The first
    stream is expected to carry no higher a rate than the second; a spike that
    arrives with the buffer already full passes straight through.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import sync_skewed

        sync = sync_skewed({'width': 3})
        first, second = sync(torch.tensor([0], dtype=torch.int8),
                             torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Stochastic Division and Square Root via Correlation*, DAC, 2019.

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*, IEEE Design & Test, 2021.
    """


    def __init__(
            self,
            config={
                'width' : 3,
            }
    ):
        """
        Configure the skew buffer counter.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Counter width in bits, giving a maximum stored skew of ``2**width - 1``; the default is ``3``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['width'], optional_key_list=['polarity'], polarity_required=False)

        #: Width of the stored stream-skew counter in bits.
        self.width=config['width']
        #: Largest unmatched-spike count retained by the synchronizer.
        self.cnt_max = 2**self.width - 1
        #: Per-element unmatched-spike count carried across timesteps.
        self.cnt: torch.Tensor
        self.register_buffer('cnt', torch.zeros(1, dtype=self.ntype))
        #: Whether :attr:`cnt` must be expanded for the first input shape.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the combinational synchronizer.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_1': 'rc', 'input_2': 'rc', 'output_1': 'rc'}
        self.polarity_io = {'input_1': 'unipolar', 'input_2': 'unipolar', 'output_1': 'unipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Clear the local skew counter and first-call shape state.
        """
        self.cnt.resize_(1).zero_()
        self.is_first_call = True


    def forward(self, input_1, input_2):
        """
        Synchronize one timestep of two input streams.

        Args:
            input_1: Current 0/1 spikes from the stream with the smaller or equal
                represented value.
            input_2: Current 0/1 spikes from the reference stream.

        Returns:
            A pair ``(output_1, input_2)``. ``output_1`` is the skew-adjusted
            first stream and the second tensor is returned unchanged. The call
            updates the saturating skew counter.

        **Example:**

        .. code-block:: python

            first, second = sync(torch.tensor([0], dtype=torch.int8),
                                 torch.tensor([1], dtype=torch.int8))
        """
        # input_1 is assumed to have no higher rate than input_2; input_2 passes through unchanged.

        # For 0/1 spikes, abs(input_1 - input_2) flags unequal pairs.
        diff = input_1 - input_2
        input_01_10 = diff.abs()
        if self.is_first_call:
            self.cnt.resize_as_(input_01_10).zero_()
            self.is_first_call = False

        cnt_not_min = torch.ne(self.cnt, 0).type(self.stype)
        cnt_not_max = torch.ne(self.cnt, self.cnt_max).type(self.stype)

        # For 0/1 input_1, select = cnt_not_min * (1 - input_1) - cnt_not_max * input_1.
        select = cnt_not_min - (cnt_not_min + cnt_not_max).mul(input_1)
        output_1 = input_1.add(input_01_10.mul(select))
        # input_01_10 * (2 * input_1 - 1) equals diff exactly; cnt remains ntype.
        self.cnt.add_(diff).clamp_(0, self.cnt_max)
        return output_1, input_2

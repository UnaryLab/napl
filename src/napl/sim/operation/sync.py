import torch

from loguru import logger
from napl.sim.base import napl_base


class sync(napl_base):
    r"""
    Raise the correlation of two rate-coded streams (SC synchronizer).

    Use this kernel before a correlation-sensitive operation such as
    :class:`~napl.sim.operation.subabs` or a maximum built from an OR gate. It
    returns two streams that carry the same values as its inputs, ``SCC`` driven
    toward ``+1``:

    .. math::

       p'_0 = p_0, \qquad p'_1 = p_1, \qquad \mathrm{SCC}(y_0,y_1) \to +1.

    Both polarities are supported. The machine rearranges spikes without
    changing either stream's rate beyond the end-of-run residue of bits still
    saved, so each stream keeps its value under the unipolar reading
    :math:`v = p` and the bipolar reading :math:`v = 2p - 1` alike.

    Each element runs one finite-state machine that dynamically pairs the ones
    and zeros of the two streams. A timestep where the two inputs agree passes
    through unchanged. A timestep where they disagree either saves the unpaired
    bit, when the machine still has room, and emits ``0`` on both outputs, or
    releases a previously saved bit of the opposite stream and emits ``1`` on
    both outputs. Once **depth** bits of one stream are saved, further unpaired
    bits of that stream pass through unchanged.

    Values are preserved only over a complete run: bits still saved when the run
    ends are never emitted, which leaves a small negative bias that shrinks as
    the stream lengthens.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import sync

        synchronizer = sync({'polarity': 'unipolar', 'depth': 1})
        first, second = synchronizer(torch.tensor([1], dtype=torch.int8),
                                     torch.tensor([0], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *Correlation Manipulating Circuits for Stochastic Computing*, DATE, 2018.
    """


    def __init__(
            self,
            config={
                'polarity' : 'unipolar',
                'depth' : 1,
            }
    ):
        """
        Configure the stream encoding and the save depth.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"unipolar"``.
              - **depth**: Number of unpaired bits of one stream the machine can save, an integer of at least ``1``; the default is ``1``. A larger depth induces stronger correlation.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'depth'], optional_key_list=[], polarity_required=True)

        #: Number of unpaired bits of one stream retained by the machine.
        self.depth = config['depth']
        if not isinstance(self.depth, int) or isinstance(self.depth, bool) or self.depth < 1:
            message = f'Invalid depth: <{self.depth}>; legal values: an integer of at least 1.'
            logger.error(message)
            raise AssertionError(message)

        # One signed counter replaces the 2 * depth + 1 chained states of the paper:
        # a positive count holds saved first-stream bits and a negative count holds
        # saved second-stream bits, and only one sign can be pending at a time.
        #: Signed count of saved bits, positive for the first stream and negative for the second.
        self.cnt: torch.Tensor
        self.register_buffer('cnt', torch.zeros(1, dtype=self.ntype))
        #: Whether :attr:`cnt` must still be expanded to the first input shape.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the combinational synchronizer outputs.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output_0': 'rc', 'output_1': 'rc'}
        self.polarity_io = {'input_0': self.polarity, 'input_1': self.polarity,
                            'output_0': self.polarity, 'output_1': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Clear the saved-bit counter and the first-call shape state.
        """
        self.cnt.resize_(1).zero_()
        self.is_first_call = True


    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor):
        """
        Synchronize one timestep of the two input streams.

        Args:
            input_0: Current 0/1 spikes from the first stream.
            input_1: Current 0/1 spikes from the second stream.

        Returns:
            A pair ``(output_0, output_1)`` of 0/1 spike tensors with the input
            shape. The call also advances the saved-bit state.

        **Example:**

        .. code-block:: python

            first, second = synchronizer(torch.tensor([1], dtype=torch.int8),
                                         torch.tensor([0], dtype=torch.int8))
        """
        is_1_0 = torch.ne(input_0, 0) & torch.eq(input_1, 0)
        is_0_1 = torch.eq(input_0, 0) & torch.ne(input_1, 0)
        if self.is_first_call:
            self.cnt.resize_as_(is_1_0.type(self.ntype)).zero_()
            self.is_first_call = False

        has_saved_0 = torch.gt(self.cnt, 0)
        has_saved_1 = torch.lt(self.cnt, 0)
        full_0 = torch.ge(self.cnt, self.depth)
        full_1 = torch.le(self.cnt, -self.depth)

        # An unpaired first-stream bit pairs with a saved second-stream bit and
        # emits 1 on both outputs, otherwise it is saved silently, and it only
        # passes through once the first-stream save slots are full.
        agree = torch.ne(input_0, 0) & torch.eq(input_0, input_1)
        output_0 = agree | (is_1_0 & (has_saved_1 | full_0)) | (is_0_1 & has_saved_0)
        output_1 = agree | (is_1_0 & has_saved_1) | (is_0_1 & (has_saved_0 | full_1))

        self.cnt.add_(is_1_0).sub_(is_0_1.type(self.ntype)).clamp_(-self.depth, self.depth)
        return output_0.type(self.stype), output_1.type(self.stype)

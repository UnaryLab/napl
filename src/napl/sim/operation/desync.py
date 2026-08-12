import torch

from loguru import logger
from napl.sim.base import napl_base


class desync(napl_base):
    r"""
    Lower the correlation of two rate-coded streams (SC desynchronizer).

    Use this kernel before an operation that needs negatively correlated
    operands, such as a saturating adder built from an OR gate. It returns two
    streams that carry the same values as its inputs, ``SCC`` driven toward
    ``-1``:

    .. math::

       p'_0 = p_0, \qquad p'_1 = p_1, \qquad \mathrm{SCC}(y_0,y_1) \to -1.

    Both polarities are supported. The machine rearranges spikes without
    changing either stream's rate, so each stream keeps its value under the
    unipolar reading :math:`v = p` and the bipolar reading :math:`v = 2p - 1`
    alike.

    Each element runs one finite-state machine that unpairs the ones of the two
    streams. A timestep where the two inputs disagree passes through unchanged,
    since it is already unpaired. A timestep where both inputs are ``1`` saves
    one of the two ones, when the machine still has room, and emits the other on
    one output only. A timestep where both inputs are ``0`` releases a saved
    ``1`` on its own output. The saved side alternates between the two streams,
    so neither output collects all the one-sided ones.

    Values are preserved only over a complete run: ones still saved when the run
    ends are never emitted, which leaves a small negative bias that shrinks as
    the stream lengthens.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import desync

        desynchronizer = desync({'polarity': 'unipolar', 'depth': 1})
        first, second = desynchronizer(torch.tensor([1], dtype=torch.int8),
                                       torch.tensor([1], dtype=torch.int8))

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
              - **depth**: Number of paired ones the machine can save on one side before alternating, an integer of at least ``1``; the default is ``1``. A larger depth induces stronger negative correlation.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'depth'], optional_key_list=[], polarity_required=True)

        #: Number of paired ones retained on one side before the saved side alternates.
        self.depth = config['depth']
        if not isinstance(self.depth, int) or isinstance(self.depth, bool) or self.depth < 1:
            message = f'Invalid depth: <{self.depth}>; legal values: an integer of at least 1.'
            logger.error(message)
            raise AssertionError(message)

        # The machine state is a signed count of saved ones in [-depth, depth] paired with the
        # side that saves next, which flips each time the count returns to zero.
        #: Signed count of saved ones, positive for the first stream and negative for the second.
        self.cnt: torch.Tensor
        self.register_buffer('cnt', torch.zeros(1, dtype=self.ntype))
        #: Side that saves the next paired one, ``+1`` for the first stream and ``-1`` for the second.
        self.side: torch.Tensor
        self.register_buffer('side', torch.ones(1, dtype=self.ntype))
        #: Whether :attr:`cnt` and :attr:`side` must still be expanded to the first input shape.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the combinational desynchronizer outputs.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output_0': 'rc', 'output_1': 'rc'}
        self.polarity_io = {'input_0': self.polarity, 'input_1': self.polarity,
                            'output_0': self.polarity, 'output_1': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Clear the saved-one counter, restore the first stream as the saving side,
        and clear the first-call shape state.
        """
        self.cnt.resize_(1).zero_()
        self.side.resize_(1).fill_(1)
        self.is_first_call = True


    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor):
        """
        Desynchronize one timestep of the two input streams.

        Args:
            input_0: Current 0/1 spikes from the first stream.
            input_1: Current 0/1 spikes from the second stream.

        Returns:
            A pair ``(output_0, output_1)`` of 0/1 spike tensors with the input
            shape. The call also advances the saved-one state.

        **Example:**

        .. code-block:: python

            first, second = desynchronizer(torch.tensor([1], dtype=torch.int8),
                                           torch.tensor([1], dtype=torch.int8))
        """
        spike_0 = torch.ne(input_0, 0)
        spike_1 = torch.ne(input_1, 0)
        both_1 = spike_0 & spike_1
        both_0 = (~spike_0) & (~spike_1)
        if self.is_first_call:
            self.cnt.resize_as_(both_1.type(self.ntype)).zero_()
            self.side.resize_as_(self.cnt).fill_(1)
            self.is_first_call = False

        save_0 = torch.gt(self.side, 0)
        room = torch.lt(self.cnt.abs(), self.depth)
        save = both_1 & room
        emit = both_0 & torch.ne(self.cnt, 0)

        # A saved one is withheld from its own output and released on that same
        # output later, so each stream keeps its value while the ones move apart.
        output_0 = (spike_0 & ~spike_1) | (both_1 & ~room) | (save & ~save_0) | (emit & torch.gt(self.cnt, 0))
        output_1 = (spike_1 & ~spike_0) | (both_1 & ~room) | (save & save_0) | (emit & torch.lt(self.cnt, 0))

        alternate = emit & torch.eq(self.cnt.abs(), 1)
        self.cnt.add_(self.side * (save.type(self.ntype) - emit.type(self.ntype)))
        self.side.mul_(torch.where(alternate, -1.0, 1.0).type(self.ntype))
        return output_0.type(self.stype), output_1.type(self.stype)

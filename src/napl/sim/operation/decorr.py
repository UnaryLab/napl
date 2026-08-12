import torch

from loguru import logger
from napl.sim.base import napl_base
from .encode import encode


class decorr(napl_base):
    r"""
    Lower the correlation of two rate-coded streams (SC decorrelator).

    Use this kernel before an operation that needs uncorrelated operands, such as
    an AND-gate multiplier fed from one shared source. It returns two streams that
    carry the same values as its inputs, ``SCC`` driven toward ``0``:

    .. math::

       p'_0 = p_0, \qquad p'_1 = p_1, \qquad \mathrm{SCC}(y_0,y_1) \to 0.

    Both polarities are supported. The buffers reorder bits without changing
    either stream's rate beyond the buffers' end-of-run residue, so each stream
    keeps its value under the unipolar reading :math:`v = p` and the bipolar
    reading :math:`v = 2p - 1` alike.

    Each stream runs its own shuffle buffer of **depth** positions, holding
    ``depth - 1`` stored bits plus one pass-through path. Every timestep an
    auxiliary number sequence picks one position: the pass-through position emits
    the current input, and a storage position emits the bit stored there and
    stores the current input in its place. The two buffers use different number
    sequences, so the two streams are reordered differently and lose their shared
    bit alignment. A larger depth scrambles bits across longer segments of the
    stream and lowers the correlation further; ``depth`` of ``1`` leaves only the
    pass-through path and returns both inputs unchanged.

    Values are preserved only over a complete run: bits still stored when the run
    ends are never emitted, and the reset contents of the buffer are emitted in
    their place, which leaves a bias bounded by ``(depth - 1) / timestep``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import decorr

        decorrelator = decorr({'polarity': 'unipolar', 'depth': 4,
                                       'timestep': 256, 'generator': 'sys'})
        first, second = decorrelator(torch.tensor([1], dtype=torch.int8),
                                     torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *Correlation Manipulating Circuits for Stochastic Computing*, DATE, 2018.
    """
    #: The buffer-position index sequences are encoded from held number
    #: sequences, so the RTL counterpart holds its own encoder instead of
    #: sharing an external one.
    internal_encode = True


    def __init__(
            self,
            config={
                'polarity' : 'unipolar',
                'depth' : 4,
                'timestep' : 256,
                'generator' : 'sys',
            }
    ):
        """
        Configure the stream encoding, the buffer depth, and the auxiliary sequence.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"unipolar"``.
              - **depth**: Number of shuffle-buffer positions, an integer of at least ``1``; the default is ``4``. A larger depth lowers the correlation further.
              - **timestep**: Period of the auxiliary number sequence that picks the buffer position; the default is ``256``.
              - **generator**: Number-sequence generator used to pick the buffer position; the default is ``"sys"``.
              - **dim**: Generator dimension forwarded when the first buffer's sequence is built; the second buffer uses the next dimension; the default is ``1``.
              - **seed**: Optional seed used when **generator** is ``"lfsr"`` or ``"sys"``; the second buffer uses the next seed; the default is ``None``.
              - **taps**: Optional LFSR feedback taps used when **generator** is ``"lfsr"``; the default is ``None``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'depth', 'timestep', 'generator'],
                         optional_key_list=['dim', 'seed', 'taps'], polarity_required=True)

        #: Number of shuffle-buffer positions, one of which is the pass-through path.
        self.depth = config['depth']
        if not isinstance(self.depth, int) or isinstance(self.depth, bool) or self.depth < 1:
            message = f'Invalid depth: <{self.depth}>; legal values: an integer of at least 1.'
            logger.error(message)
            raise AssertionError(message)

        # Each buffer needs its own random source, so the two sequences differ by
        # generator dimension for Sobol and by seed for the seeded generators.
        seed = config.get('seed', None)
        #: Per-buffer lists of buffer positions selected by the configured number sequences.
        self.rand_seq_idx = [
            encode({'polarity': 'unipolar',
                    'timestep': config['timestep'],
                    'generator': config['generator'],
                    'dim': config.get('dim', 1) + offset,
                    'seed': None if seed is None else int(seed) + offset,
                    'taps': config.get('taps', None)}).num_seq.mul(self.depth).type(torch.long).tolist()
            for offset in range(2)
        ]

        #: Stored bits of both shuffle buffers, indexed by stream and buffer position.
        self.reg: torch.Tensor
        self.register_buffer('reg', torch.zeros((2, self.depth - 1), dtype=self.stype))
        for slot in range(self.depth - 1):
            self.reg[:, slot].fill_(slot % 2)
        #: Whether :attr:`reg` must still be expanded to the first input shape.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the combinational multiplexer output.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output_0': 'rc', 'output_1': 'rc'}
        self.polarity_io = {'input_0': self.polarity, 'input_1': self.polarity,
                            'output_0': self.polarity, 'output_1': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the alternating buffer contents and the first-call shape state.
        """
        self.reg.resize_((2, self.depth - 1)).zero_()
        for slot in range(self.depth - 1):
            self.reg[:, slot].fill_(slot % 2)
        self.is_first_call = True


    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor):
        """
        Decorrelate one timestep of the two input streams.

        Args:
            input_0: Current 0/1 spikes from the first stream.
            input_1: Current 0/1 spikes from the second stream.
                Both inputs share one shape.

        Returns:
            A pair ``(output_0, output_1)`` of 0/1 spike tensors with the input
            shape. The call also updates the shuffle buffers, whose selected
            position is given by ``rand_seq_idx`` indexed by ``timestep_cur``.
            The buffer shape is fixed by the first call after ``reset()``; a
            different input shape requires ``reset()``.

        **Example:**

        .. code-block:: python

            first, second = decorrelator(torch.tensor([1], dtype=torch.int8),
                                         torch.tensor([1], dtype=torch.int8))
        """
        if self.is_first_call:
            self.reg.resize_((2, self.depth - 1, *input_0.shape))
            for slot in range(self.depth - 1):
                self.reg[:, slot].fill_(slot % 2)
            self.is_first_call = False

        output = []
        for stream, input in enumerate((input_0, input_1)):
            sequence = self.rand_seq_idx[stream]
            index = sequence[(self.timestep_cur - 1) % len(sequence)]
            # The last position is the multiplexer's pass-through input, which has
            # no storage cell behind it.
            if index == self.depth - 1:
                output.append(input.type(self.stype).clone())
            else:
                slot = self.reg[stream, index]
                output.append(slot.clone())
                slot.copy_(input)
        return output[0], output[1]

import torch
from loguru import logger

from napl.sim.base import napl_base
from .encode import encode


class encode_regen(napl_base):
    r"""Regenerate a short-budget Sobol-encoded unipolar input stream.

    A ``window``-bit registered history supplies a popcount :math:`k`. One
    private extended LFSR supplies an integer threshold :math:`r`, and the output
    is one exactly when :math:`r < k`.

    A bare maximal LFSR visits only the nonzero states, so modulo a power-of-two
    window it assigns threshold zero one fewer draw than every other threshold.
    The shared ``lfsr_ext`` generator inserts the missing zero state at the
    standard all-zero run transition, extending the cycle to the
    :math:`2^L`-state de Bruijn cycle. Every threshold then occurs exactly
    :math:`2^L/W` times. The extension needs only detection of the insertion
    state in hardware; it does not add a ROM.

    Uniform thresholds do not make the finite-window estimate independent of
    its input. The regular input stream, overlapping window, and periodic
    ``lfsr_ext`` threshold sequence retain deterministic correlation and
    bias. The focused test prints the Sobol fidelity and input/output SCC for
    the supported design points.

    Only unipolar streams are supported because the window popcount estimates
    a unipolar rate directly; bipolar regeneration needs a recentered estimate.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import encode_regen

        regenerate = encode_regen({'polarity': 'unipolar'})
        output = regenerate(torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        Derived from *Stochastic Computing Systems*, Advances in Information Systems Science, 1969.
    """


    #: Smallest supported private-LFSR width.
    LFSR_WIDTH_MIN = 3
    #: Largest supported private-LFSR width.
    LFSR_WIDTH_MAX = 8
    #: Largest supported rate-estimation window.
    WINDOW_MAX = 8
    #: Dominant hardware mechanism of this class.
    mechanism = 'regeneration'


    def __init__(
            self,
            config={
                'polarity': 'unipolar',
                'lfsr_width': 3,
                'window': 4,
            }
        ):
        """Configure the registered window and private extended LFSR.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** - Configuration mapping.

              - **polarity**: Must be ``"unipolar"``.
              - **lfsr_width**: Private maximal-LFSR width from ``3`` to ``8``;
                the default is ``3``.
              - **window**: Power-of-two history length from ``2`` to ``8``;
                the default is ``4``.
              - **name**: Optional instance label.
        """
        super().__init__(
            config,
            ['polarity'],
            optional_key_list=['lfsr_width', 'window'],
            polarity_required=True,
        )

        if self.polarity != 'unipolar':
            message = f'Invalid polarity: <{self.polarity}>; encode_regen supports unipolar only.'
            logger.error(message)
            raise AssertionError(message)

        #: Width of the private extended LFSR.
        self.lfsr_width = config.get('lfsr_width', 3)
        if (type(self.lfsr_width) is not int
                or not self.LFSR_WIDTH_MIN <= self.lfsr_width <= self.LFSR_WIDTH_MAX):
            message = (
                f'Invalid lfsr_width: <{self.lfsr_width}>; legal values: '
                f'an integer from {self.LFSR_WIDTH_MIN} to {self.LFSR_WIDTH_MAX}.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Number of spikes in the registered rate-estimation window.
        self.window = config.get('window', 4)
        if (type(self.window) is not int or not 2 <= self.window <= self.WINDOW_MAX
                or self.window & (self.window - 1)):
            message = (
                f'Invalid window: <{self.window}>; legal values: '
                f'a power-of-two integer from 2 to {self.WINDOW_MAX}.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Encoder owning the private extended maximal-LFSR threshold sequence.
        self.reference_encode = encode({
            'polarity': 'unipolar',
            'timestep': 2 ** self.lfsr_width,
            'generator': 'lfsr_ext',
        })
        # Dividing by the power-of-two period is exact, so the encoder sequence
        # scaled back to integers reproduces the extended-LFSR states exactly.
        # The comparator itself stays inline: it folds the state modulo the
        # window, and the encoder compares against the unfolded sequence, so a
        # per-timestep encoder call would only agree when window equals the
        # sequence period.
        states = self.reference_encode.num_seq.mul(2 ** self.lfsr_width).round().type(torch.long)
        self._lfsr_period = 2 ** self.lfsr_width
        #: Integer states of the private extended maximal-LFSR cycle.
        self.lfsr_states: torch.Tensor
        self.register_buffer('lfsr_states', states)
        #: Current position in the extended-LFSR state cycle.
        self.lfsr_index: torch.Tensor
        self.register_buffer('lfsr_index', torch.zeros((), dtype=torch.long))
        #: Registered input window, expanded to the input tensor shape on first use.
        self.window_register: torch.Tensor
        self.register_buffer('window_register', torch.zeros(self.window, dtype=self.stype))
        self.register_buffer('_initialized', torch.tensor(False))

        # The selected design point is width 3 with window 4.
        #: Hardware latency for the registered input window read by the comparator.
        self.hw.pp_delay = 1
        #: Encoder the hardware counterpart carries. The private extended LFSR
        #: that supplies the threshold belongs to this operation alone.
        self.internal_encode = 'private'
        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'unipolar', 'output': 'unipolar'}
        self.correlation_i = {}
        self.correlation_o = {}


    def _reset(self):
        """Restore the empty input window and opening LFSR position."""
        self.window_register.resize_(self.window).zero_()
        self.lfsr_index.zero_()
        self._initialized.fill_(False)


    def forward(self, input: torch.Tensor):
        """Regenerate one input-spike timestep from the registered rate estimate.

        Args:
            input: Current unipolar 0/1 spike tensor.

        Returns:
            A same-shaped unipolar spike tensor. The window and LFSR position
            advance once. The first call fills the window with that input so
            constant-zero and constant-one streams retain their exact values.
        """
        input_stype = input.type(self.stype)
        first_call = not bool(self._initialized)
        if first_call:
            self.window_register.resize_(self.window, *input.shape)
            self.window_register.copy_(
                input_stype.unsqueeze(0).expand_as(self.window_register)
            )
            self._initialized.fill_(True)

        popcount = self.window_register.sum(dim=0)
        state = self.lfsr_states[self.lfsr_index]
        output = state.remainder(self.window).lt(popcount).type(self.stype)
        self.lfsr_index.add_(1).remainder_(self._lfsr_period)
        if not first_call:
            head = (self.timestep_cur - 1) % self.window
            self.window_register[head].copy_(input_stype)
        return output

import torch
from loguru import logger

from napl.sim.base import napl_base
from .encode_regen import encode_regen
from .mul_gaines import mul_gaines


class pow_regen(napl_base):
    r"""
    Raise a unipolar rate-coded stream to a small integer power by regeneration.

    A ``window``-bit input history estimates the current rate. One private
    extended LFSR supplies a uniform integer threshold, and comparison with the
    history popcount regenerates a stream at that estimated rate. The output is
    the Gaines product of ``n`` copies of the regenerated stream spaced
    ``depth`` timesteps apart, including the current regenerated bit.

    The default ``(lfsr_width, window, depth) = (3, 4, 1)`` is the selected
    small-area point. The delayed regenerated factors violate the
    zero-correlation relation required by ``mul_gaines``, and those depth-tap
    correlations drive the default configuration's residual error. For
    configurations of the ``(8, 8, 3)`` class, the residual error comes from
    the ``encode_regen`` rate bias and from phase lock between a Sobol-encoded
    input and the periodic ``lfsr_ext`` threshold sequence. Changing the depth
    changes the factors' phase interaction. Callers should use Sobol-encoded
    input and inspect the focused test's printed fidelity and
    factor-correlation evidence for their operating point.

    Only unipolar streams are supported; no bipolar XNOR regeneration path is
    provided.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import pow_regen

        cube = pow_regen({'polarity': 'unipolar', 'n': 3})
        output = cube(torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        Derived from *Stochastic Computing Systems*, Advances in Information Systems Science, 1969.
    """


    #: Largest supported integer power.
    N_MAX = 8
    #: Smallest supported private-LFSR width.
    LFSR_WIDTH_MIN = encode_regen.LFSR_WIDTH_MIN
    #: Largest supported private-LFSR width.
    LFSR_WIDTH_MAX = encode_regen.LFSR_WIDTH_MAX
    #: Largest supported input-history window.
    WINDOW_MAX = encode_regen.WINDOW_MAX
    #: Dominant hardware mechanism of this class.
    mechanism = 'regeneration'


    def __init__(
            self,
            config={
                'polarity': 'unipolar',
                'n': 2,
                'lfsr_width': 3,
                'window': 4,
                'depth': 1,
            }
        ):
        """Configure the power, regenerated-rate window, and private LFSR.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** - Configuration mapping.

              - **polarity**: Must be ``"unipolar"``.
              - **n**: Integer power from ``2`` to ``8``; the default is ``2``.
              - **lfsr_width**: Private extended-LFSR width from ``3`` to ``8``;
                the default is ``3``.
              - **window**: Power-of-two history length from ``2`` to ``8``;
                the default is ``4``.
              - **depth**: Positive spacing between regenerated factors; the
                default is ``1``.
              - **name**: Optional instance label.
        """
        super().__init__(
            config,
            ['polarity', 'n'],
            optional_key_list=['lfsr_width', 'window', 'depth'],
            polarity_required=True,
        )

        if self.polarity != 'unipolar':
            message = f'Invalid polarity: <{self.polarity}>; pow_regen supports unipolar only.'
            logger.error(message)
            raise AssertionError(message)

        #: Integer power the input stream is raised to.
        self.n = config['n']
        if type(self.n) is not int or not 2 <= self.n <= self.N_MAX:
            message = f'Invalid n: <{self.n}>; legal values: an integer from 2 to {self.N_MAX}.'
            logger.error(message)
            raise AssertionError(message)

        #: Timestep spacing between consecutive regenerated factors.
        self.depth = config.get('depth', 1)
        if type(self.depth) is not int or self.depth < 1:
            message = f'Invalid depth: <{self.depth}>; legal values: an integer of at least 1.'
            logger.error(message)
            raise AssertionError(message)

        self._depth_span = (self.n - 1) * self.depth
        #: Held window/popcount/LFSR comparator that regenerates the input stream.
        self.regenerator = encode_regen({
            'polarity': self.polarity,
            'lfsr_width': config.get('lfsr_width', 3),
            'window': config.get('window', 4),
        })
        #: Width of the held private extended LFSR.
        self.lfsr_width = self.regenerator.lfsr_width
        #: Number of input spikes in the held rate-estimation window.
        self.window = self.regenerator.window
        #: Registered regenerated-bit depth history, expanded on first use.
        self.depth_register: torch.Tensor
        self.register_buffer(
            'depth_register', torch.zeros(max(1, self._depth_span), dtype=self.stype),
        )
        self.register_buffer('_depth_initialized', torch.tensor(False))

        multipliers = [
            mul_gaines({'polarity': 'unipolar'}) for _ in range(self.n - 1)
        ]
        for index, operation in enumerate(multipliers):
            self.add_module(f'multiplier_{index}', operation)
        #: Held Gaines multipliers for the current and prior regenerated factors.
        self.multipliers = tuple(multipliers)

        # The selected design point is width 3, window 4, and depth 1.
        #: Hardware latency inherited from the registered-window regenerator.
        self.hw.pp_delay = self.regenerator.hw.pp_delay
        #: Encoder the hardware counterpart carries, derived from the registered
        #: parts: ``'private'`` when any part carries an encoder of its own,
        #: ``'none'`` otherwise.
        self.internal_encode = 'private' if any(part.internal_encode != 'none' for part in self.children()) else 'none'
        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'unipolar', 'output': 'unipolar'}
        self.correlation_i = {}
        self.correlation_o = {}


    def _reset(self):
        """Restore the local regenerated-bit depth history to its empty state."""
        self.depth_register.resize_(max(1, self._depth_span)).zero_()
        self._depth_initialized.fill_(False)


    def forward(self, input: torch.Tensor):
        """Process one input-spike timestep and return its regenerated power.

        Args:
            input: Current unipolar 0/1 spike tensor.

        Returns:
            A same-shaped unipolar spike tensor. The held regenerator and local
            regenerated-bit depth history advance once.
        """
        input_stype = input.type(self.stype)
        regenerated = self.regenerator(input_stype)
        first_call = not bool(self._depth_initialized)
        if first_call and self._depth_span:
            self.depth_register.resize_(self._depth_span, *input.shape)
        self._depth_initialized.fill_(True)

        factors = [regenerated]
        if self._depth_span:
            if first_call:
                self.depth_register.copy_(
                    regenerated.unsqueeze(0).expand_as(self.depth_register)
                )
                factors.extend([regenerated] * (self.n - 1))
            else:
                head = (self.timestep_cur - 2) % self._depth_span
                for tap_depth in range(self.depth, self._depth_span + 1, self.depth):
                    index = (head + self._depth_span - tap_depth) % self._depth_span
                    factors.append(self.depth_register[index].clone())
                self.depth_register[head].copy_(regenerated)

        output = factors[0]
        for multiplier, factor in zip(self.multipliers, factors[1:]):
            output = multiplier(output, factor)
        return output

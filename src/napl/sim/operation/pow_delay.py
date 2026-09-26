import torch
from loguru import logger

from napl.sim.base import napl_base
from .delay import delay
from .mul_gaines import mul_gaines


class pow_delay(napl_base):
    r"""
    Raise a unary stream to an integer power using delayed copies.

    The operation multiplies the current spike with :math:`n-1` copies spaced
    by ``depth`` timesteps. It is the minimum-area power variant, holding
    exactly :math:`(n-1)\,\mathit{depth}` one-bit delay registers and using no
    generated number sequence or memory. Use :class:`pow_regen` when numerical
    accuracy matters more than this minimum state.

    The delayed factors violate the zero-correlation input relation
    :class:`mul_gaines` requires by design, and a zero-output dead zone is
    the consequence: under the Sobol encoder, higher powers return exactly
    zero below an input boundary that depends on the number of delayed
    factors. At high input rates, Sobol lag anti-correlation drives the
    joint-one rate to its Frechet lower bound. The focused test prints the
    per-power dead-zone boundaries and the generator sensitivity.

    The default ``depth=1`` is the selected minimum-state point. The depth
    changes accuracy, and the focused test prints the depth sweep. The same
    test reports the bipolar behavior.

    The power, generator, and depth change the input/output SCC, and the
    focused test prints the SCC across those parameters. The class therefore
    declares no unconditional output-correlation relation.

    The first :math:`(n-1)\,\mathit{depth}` results use at least one factor
    from the zero-filled delay line. Both unipolar (AND) and bipolar (XNOR)
    streams are supported over their full legal ranges.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import pow_delay

        cube = pow_delay({'polarity': 'bipolar', 'n': 3})
        output = cube(torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        Derived from *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.

        Derived from *Stochastic Computing Systems*, Advances in Information Systems Science, 1969.
    """


    #: Largest supported power.
    N_MAX = 8
    #: Dominant hardware mechanism of this class.
    mechanism = 'delay'


    def __init__(
            self,
            config={
                'polarity': 'bipolar',
                'n': 2,
                'depth': 1,
            }
        ):
        """
        Configure the stream polarity and integer power.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Use ``"unipolar"`` for AND or ``"bipolar"`` for XNOR; the default is ``"bipolar"``.
              - **n**: Integer power from ``2`` to ``8``; the default is ``2``.
              - **depth**: Positive timestep spacing between factors; the default is ``1``. There is no upper bound; a larger depth only grows the delay line.
              - **name**: Optional instance label.
        """
        super().__init__(
            config, ['polarity', 'n'], optional_key_list=['depth'],
            polarity_required=True,
        )

        #: Integer power the input stream is raised to.
        self.n = config['n']
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 2 or self.n > self.N_MAX:
            message = f'Invalid n: <{self.n}>; legal values: an integer from 2 to {self.N_MAX}.'
            logger.error(message)
            raise AssertionError(message)

        #: Timestep spacing between consecutive delayed factors.
        self.depth = config.get('depth', 1)
        if type(self.depth) is not int or self.depth < 1:
            message = (
                f'Invalid depth: <{self.depth}>; legal values: '
                'an integer of at least 1.'
            )
            logger.error(message)
            raise AssertionError(message)

        # The selected depth-1 area point is non-dominated: at n=3 and T=256 it
        # measures 0.064762/0.179537 mean/max error using 2 FFs, no ROM, and 2
        # gates, while pow_regen measures 0.038283/0.103847 using 9 FFs, no ROM,
        # and about 8 LUT6 equivalents. General pow_delay cost is
        # (n-1)*depth FFs, no memory, and n-1 gates at logic depth n-1.
        delays = [delay({'depth': self.depth}) for _ in range(self.n - 1)]
        multipliers = [
            mul_gaines({'polarity': self.polarity}) for _ in range(self.n - 1)
        ]
        for index, operation in enumerate(delays):
            self.add_module(f'delay_{index}', operation)
        for index, operation in enumerate(multipliers):
            self.add_module(f'multiplier_{index}', operation)
        #: Cascaded delay stages whose taps provide the spaced earlier factors.
        self.delays = tuple(delays)
        #: Gaines multipliers that combine the current and delayed factors.
        self.multipliers = tuple(multipliers)

        # The current input reaches the output through only combinational
        # multipliers; the registered taps represent earlier input samples.
        #: Hardware latency and timing metadata for the combinational power output.
        self.hw.pp_delay = 0

        #: Encoder the hardware counterpart carries, derived from the registered
        #: parts: ``'private'`` when any part carries an encoder of its own,
        #: ``'none'`` otherwise.
        self.internal_encode = 'private' if any(part.internal_encode != 'none' for part in self.children()) else 'none'
        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.correlation_o = {}


    def _reset(self):
        """
        Reset no class-local state beyond the registered delay stages.
        """
        pass


    def forward(self, input: torch.Tensor):
        """
        Multiply the current spike by its preceding ``n - 1`` spikes.

        Args:
            input: Current 0/1 spike tensor.

        Returns:
            The bitwise delayed product in the configured polarity. Each delay
            stage stores its input for the next timestep.

        **Example:**

        .. code-block:: python

            output = cube(torch.tensor([1], dtype=torch.int8))
        """
        factors = [input]
        delayed = input
        for stage in self.delays:
            delayed = stage(delayed)
            factors.append(delayed)

        output = factors[0]
        for factor, multiplier in zip(factors[1:], self.multipliers):
            output = multiplier(output, factor)
        return output

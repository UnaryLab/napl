import torch
from loguru import logger

from napl.sim.base import napl_base
from .dff import dff
from .mul_gaines import mul_gaines


class pow_n(napl_base):
    r"""
    Raise a unary stream to a small integer power :math:`n`.

    The rate-domain operation is

    .. math::

       p_y = p_x^n \quad (\text{unipolar}),\qquad
       v_y = v_x^n \quad (\text{bipolar}).

    This is the :class:`~napl.sim.operation.square_dff` construction generalized
    from :math:`x^2` to :math:`x^n`: the stream is multiplied against
    :math:`n-1` decorrelated (delayed) copies of itself, one Gaines multiply per
    copy. Each extra multiply compounds the stochastic-computing variance, so the
    decoded error grows with :math:`n`; expect a visibly larger error than a
    single multiply, and larger still as :math:`n` rises. Because of that growth
    :math:`n` is capped at ``8``, past which a fixed-length stream no longer
    resolves the power usefully.

    Both polarities of the underlying Gaines multiplier are supported: unipolar
    (AND) and bipolar (XNOR). The legal input range is :math:`[0, 1]` unipolar
    and :math:`[-1, 1]` bipolar; :math:`x^n` stays in range for either polarity.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import pow_n

        cube = pow_n({'polarity': 'bipolar', 'n': 3})
        output = cube(torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        Derived from *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.

        Derived from *Stochastic Computing Systems*, Advances in Information Systems Science, 1969.
    """


    #: Largest supported power; beyond this a fixed-length stream cannot resolve the power.
    N_MAX = 8


    def __init__(
            self,
            config={
                'polarity': 'bipolar',
                'n': 2,
            }
        ):
        """
        Configure the stream polarity and the integer power.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Use ``"unipolar"`` for AND or ``"bipolar"`` for XNOR; the default is ``"bipolar"``.
              - **n**: Integer power to raise the stream to, from ``2`` to ``8``; the default is ``2``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'n'], polarity_required=True)

        #: Integer power the input stream is raised to.
        self.n = config['n']
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 2 or self.n > self.N_MAX:
            message = f'Invalid n: <{self.n}>; legal values: an integer from 2 to {self.N_MAX}.'
            logger.error(message)
            raise AssertionError(message)

        # x^n is a chain of n-1 Gaines multiplies. The multiplier is stateless, so
        # one instance is reused for every stage; each stage instead needs its own
        # decorrelated copy of the input, produced by a dff whose delay is unique to
        # that stage (depths 1..n-1). Distinct delays give distinct stream phases, so
        # the n factors x, dff_1(x), ..., dff_{n-1}(x) are mutually decorrelated, which
        # is what the Gaines multiply requires. Equal delays would repeat a phase and
        # collapse the product back toward x^2.
        self.mul = mul_gaines({'polarity': self.polarity})
        #: One delay line per multiply stage, each holding a differently phased copy of the input.
        self.dffs = torch.nn.ModuleList(dff({'depth': k}) for k in range(1, self.n))

        # pp_delay sums the stage delays: stage k contributes its dff depth k and the
        # combinational multiplier adds 0, so the chain latency is 1+2+...+(n-1).
        #: Hardware latency and timing metadata for the chained power output.
        self.hw.pp_delay = sum(range(1, self.n))

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Reset the per-stage delay lines; the stateless multiplier holds nothing.
        """
        # The delay lines live in a ModuleList, whose own object has no reset(), so
        # napl_base.reset() does not recurse into them; reset each stage explicitly.
        for delay in self.dffs:
            delay.reset()


    def forward(self, input: torch.Tensor):
        """
        Raise one timestep to the configured power.

        Args:
            input: Current 0/1 spike tensor.

        Returns:
            The stream raised to power ``n`` for this timestep, in the configured
            spike dtype. Each internal delay line stores the current input for its
            later timesteps.

        **Example:**

        .. code-block:: python

            output = cube(torch.tensor([1], dtype=torch.int8))
        """
        out = input
        for delay in self.dffs:
            out = self.mul(out, delay(input))
        return out

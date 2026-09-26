import torch
from loguru import logger

from napl.sim.base import napl_base
from .encode_regen import encode_regen
from .delay import delay
from .mul_gaines import mul_gaines
from .mul_scale import mul_scale


class exp_m1_regen(napl_base):
    r"""
    Approximate :math:`\exp(x-1)` from a unipolar rate-coded stream.

    Use this streaming kernel for inputs in :math:`[0, 1]`. With
    :math:`u=1-x`, it evaluates the order-``N`` Horner form

    .. math::

       1-u\left(1-\frac{u}{2}\left(1-\frac{u}{3}
       \left(\cdots\left(1-\frac{u}{N}\right)\right)\right)\right),

    which truncates the Taylor series of :math:`\exp(-u)` after order ``N``.
    The truncation error is one-sided and bounded by :math:`1/(N+1)!`.

    Each Horner product multiplies the current scaled ``u/k`` stream by a
    regenerated, depth-delayed copy of the nested-stage stream. The innermost
    ``1-u/N`` seed is combinational, and the ``k=1`` product uses raw ``u``.
    The scaled operand stays combinational from the current input. Regeneration
    uses one private ``encode_regen`` per product so the Gaines product does not
    multiply two streams taken from the same current-input path. The kernel
    holds :math:`N-1` private ``encode_regen`` units,
    :math:`(N-1)\,\mathit{depth}` delay registers, :math:`N-1` Gaines
    multipliers, and :math:`N-1` scalers for :math:`k=2` through :math:`N`.
    Only unipolar streams are supported.

    The first :math:`(N-1)\,\mathit{depth}` results use at least one factor
    from a zero-filled delay tap.

    The focused test prints the operand-comparison evidence for regenerating
    the nested-stage stream rather than the scaled stream, together with the
    Sobol fidelity against :math:`\exp(x-1)`.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import exp_m1_regen

        operation = exp_m1_regen({'polarity': 'unipolar', 'order': 5})
        output = operation(torch.tensor([0, 1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        Derived from *Computing Arithmetic Functions Using Stochastic Logic by Series Expansion*, IEEE Transactions on Emerging Topics in Computing, 2019.
    """


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
                'order': 5,
                'lfsr_width': 3,
                'window': 4,
                'depth': 1,
            }
        ):
        """Configure the unipolar Taylor series and its per-stage regenerators.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** - Configuration mapping.

              - **polarity**: Must be ``"unipolar"``; the default is ``"unipolar"``.
              - **order**: Taylor-series order, an integer of at least ``1``;
                the default is ``5``.
              - **lfsr_width**: Private extended-LFSR width from ``3`` to ``8``;
                the default is ``3``.
              - **window**: Power-of-two history length from ``2`` to ``8``;
                the default is ``4``.
              - **depth**: Positive delay on each regenerated nested-stage
                stream; the default is ``1``.
              - **name**: Optional instance label.
        """
        super().__init__(
            config,
            ['polarity', 'order'],
            optional_key_list=['lfsr_width', 'window', 'depth'],
            polarity_required=True,
        )
        if self.polarity != 'unipolar':
            message = (
                f'Invalid polarity: <{self.polarity}>; '
                f'exp_m1_regen supports unipolar only.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Taylor-series order used by the Horner chain.
        self.order = config['order']
        if type(self.order) is not int or self.order < 1:
            message = (
                f'Invalid order: <{self.order}>; '
                f'legal values: an integer of at least 1.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Timestep delay applied to each regenerated nested-stage stream.
        self.depth = config.get('depth', 1)
        if type(self.depth) is not int or self.depth < 1:
            message = (
                f'Invalid depth: <{self.depth}>; '
                f'legal values: an integer of at least 1.'
            )
            logger.error(message)
            raise AssertionError(message)

        scalers = [
            mul_scale({
                'polarity': 'unipolar',
                'scale': 1 / k,
                'intwidth': 8,
                'fracwidth': 12,
            })
            for k in range(2, self.order + 1)
        ]
        n_prod = max(0, self.order - 1)
        regenerators = [
            encode_regen({
                'polarity': 'unipolar',
                'lfsr_width': config.get('lfsr_width', 3),
                'window': config.get('window', 4),
            })
            for _ in range(n_prod)
        ]
        delays = [delay({'depth': self.depth}) for _ in range(n_prod)]
        multipliers = [
            mul_gaines({'polarity': 'unipolar'}) for _ in range(n_prod)
        ]
        for index, operation in enumerate(scalers):
            self.add_module(f'scaler_{index}', operation)
        for index, operation in enumerate(regenerators):
            self.add_module(f'regenerator_{index}', operation)
        for index, operation in enumerate(delays):
            self.add_module(f'delay_{index}', operation)
        for index, operation in enumerate(multipliers):
            self.add_module(f'multiplier_{index}', operation)
        #: Fixed-point scalers producing the ``u / k`` stream for stages 2 through N.
        self.scalers = tuple(scalers)
        #: Private regenerators that rebuild each nested-stage stream.
        self.regenerators = tuple(regenerators)
        #: Depth delays on each regenerated nested-stage stream.
        self.delays = tuple(delays)
        #: Unipolar Gaines multipliers for the Horner products.
        self.multipliers = tuple(multipliers)
        if self.regenerators:
            #: Width of each private extended LFSR.
            self.lfsr_width = self.regenerators[0].lfsr_width
            #: Number of input spikes in each regenerator's rate-estimation window.
            self.window = self.regenerators[0].window
        else:
            #: Width of each private extended LFSR.
            self.lfsr_width = config.get('lfsr_width', 3)
            if (type(self.lfsr_width) is not int
                    or not self.LFSR_WIDTH_MIN <= self.lfsr_width <= self.LFSR_WIDTH_MAX):
                message = (
                    f'Invalid lfsr_width: <{self.lfsr_width}>; legal values: '
                    f'an integer from {self.LFSR_WIDTH_MIN} to {self.LFSR_WIDTH_MAX}.'
                )
                logger.error(message)
                raise AssertionError(message)
            #: Number of input spikes in each regenerator's rate-estimation window.
            self.window = config.get('window', 4)
            if (type(self.window) is not int or not 2 <= self.window <= self.WINDOW_MAX
                    or self.window & (self.window - 1)):
                message = (
                    f'Invalid window: <{self.window}>; legal values: '
                    f'a power-of-two integer from 2 to {self.WINDOW_MAX}.'
                )
                logger.error(message)
                raise AssertionError(message)

        # RULE_IMP gate 10 design point: regenerate and delay the nested-stage
        # operand; scaled u/k stays combinational. The selected point uses
        # N-1 regenerators, (N-1)*depth FFs, and N-1 Gaines gates, not N of
        # each. Sobol unipolar T=256, order=5, (3, 4, 1), linspace(0, 1, 128):
        # A nested RMSE=0.01219764 max_abs=0.03470451; B scaled RMSE=0.05017024
        # max_abs=0.12416479.
        #: Hardware latency for the combinational current-input-to-output path.
        self.hw.pp_delay = 0

        #: Encoder the hardware counterpart carries, derived from the registered
        #: parts: ``'private'`` when any part carries an encoder of its own,
        #: ``'none'`` otherwise.
        self.internal_encode = 'private' if any(part.internal_encode != 'none' for part in self.children()) else 'none'
        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'unipolar', 'output': 'unipolar'}
        self.correlation_i = {}
        self.correlation_o = {}


    def _reset(self):
        """Reset no class-local state beyond the registered child operations."""
        pass


    def forward(self, input: torch.Tensor):
        """Evaluate one timestep of the Taylor-series approximation.

        Args:
            input: Current tensor of unipolar 0/1 spikes.

        Returns:
            A unipolar output spike tensor with the same shape and spike dtype.
        """
        # No operation owns unipolar complementation, so NOT is expressed directly.
        u = 1 - input.type(torch.int8)
        if self.order == 1:
            return (1 - u).type(self.stype)
        stage = 1 - self.scalers[-1](u)
        for k in range(self.order - 1, 0, -1):
            scaled = u if k == 1 else self.scalers[k - 2](u)
            # Each product regenerates and depth-delays the nested operand so the
            # Gaines product sees a current scaled u/k stream and a delayed
            # regenerated nested-stage stream.
            regenerated = self.regenerators[k - 1](stage)
            delayed = self.delays[k - 1](regenerated)
            # No operation owns unipolar complementation, so the Horner NOT is direct.
            stage = 1 - self.multipliers[k - 1](scaled, delayed)
        return stage.type(self.stype)

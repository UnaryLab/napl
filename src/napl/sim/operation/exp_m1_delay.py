import torch
from loguru import logger

from napl.sim.base import napl_base
from .delay import delay
from .mul_gaines import mul_gaines
from .mul_scale import mul_scale


class exp_m1_delay(napl_base):
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
    delayed copy of the nested-stage stream. The innermost factor is the
    combinational ``1-u/N`` stream; at ``k=1`` the scaled operand is raw
    ``u``. The nested operand is the delayed factor so the current
    ``u/k`` path stays combinational and ``pp_delay`` is ``0``. Delayed
    factors violate the zero-correlation input relation
    :class:`mul_gaines` requires; the focused test prints the consequence.

    The default ``depth=1`` is the selected minimum-state point. The kernel
    holds :math:`(N-1)\,\mathit{depth}` one-bit delay registers, :math:`N-1`
    Gaines multipliers, and :math:`N-1` scalers for :math:`k=2` through
    :math:`N`, and generates no number sequence. Depth and order change
    accuracy and SCC, and the focused test prints those measurements.

    The first :math:`(N-1)\,\mathit{depth}` results use at least one factor
    from a zero-filled delay tap. For ``order`` above 1, the first ``depth`` outputs are 1
    regardless of input, because every product sees a zero-filled tap.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import exp_m1_delay

        operation = exp_m1_delay({
            'polarity': 'unipolar', 'order': 5,
        })
        output = operation(torch.tensor([0, 1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        Derived from *Computing Arithmetic Functions Using Stochastic Logic by Series Expansion*, IEEE Transactions on Emerging Topics in Computing, 2019.
    """
    #: Dominant hardware mechanism of this class.
    mechanism = 'delay'


    def __init__(
            self,
            config={
                'polarity': 'unipolar',
                'order': 5,
                'depth': 1,
            }
        ):
        """
        Configure the unipolar Taylor series and its delay taps.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding. The only supported value is ``"unipolar"``; the default is ``"unipolar"``.
              - **order**: Taylor-series order, an integer of at least ``1``; the default is ``5``.
              - **depth**: Positive timestep delay of each nested Horner operand; the default is ``1``. There is no upper bound; a larger depth only grows the delay line.
              - **name**: Optional instance label.
        """
        super().__init__(
            config, ['polarity', 'order'], optional_key_list=['depth'],
            polarity_required=True,
        )
        if self.polarity != 'unipolar':
            message = f'Invalid polarity: <{self.polarity}>; exp_m1_delay supports unipolar only.'
            logger.error(message)
            raise AssertionError(message)

        #: Taylor-series order used by the Horner chain.
        self.order = config['order']
        if type(self.order) is not int or self.order < 1:
            message = f'Invalid order: <{self.order}>; legal values: an integer of at least 1.'
            logger.error(message)
            raise AssertionError(message)

        #: Timestep delay of each nested Horner operand.
        self.depth = config.get('depth', 1)
        if type(self.depth) is not int or self.depth < 1:
            message = (
                f'Invalid depth: <{self.depth}>; legal values: '
                'an integer of at least 1.'
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
        # Depth-1 is the selected minimum-state design point.
        delays = [delay({'depth': self.depth}) for _ in range(max(0, self.order - 1))]
        multipliers = [
            mul_gaines({'polarity': 'unipolar'})
            for _ in range(max(0, self.order - 1))
        ]
        for index, operation in enumerate(scalers):
            self.add_module(f'scaler_{index}', operation)
        for index, operation in enumerate(delays):
            self.add_module(f'delay_{index}', operation)
        for index, operation in enumerate(multipliers):
            self.add_module(f'multiplier_{index}', operation)
        #: Fixed-point scalers producing the ``u / k`` stream for ``k`` from 2 through ``N``.
        self.scalers = tuple(scalers)
        #: Delay stages that hold the nested Horner operand of each product.
        self.delays = tuple(delays)
        #: Unipolar Gaines multipliers for the Horner products.
        self.multipliers = tuple(multipliers)

        # The current u/k path reaches the output through only combinational
        # scalers, multipliers, and NOT; the registered taps hold earlier nested
        # stage samples.
        #: Hardware latency and timing metadata for the combinational Horner output.
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
        """
        Evaluate one timestep of the Taylor-series approximation.

        Args:
            input: Current tensor of unipolar 0/1 spikes.

        Returns:
            A unipolar output spike tensor with the same shape and spike dtype.
            Each delay stage stores its nested operand for the next timestep.
        """
        # No operation owns unipolar complementation, so NOT is expressed directly.
        u = 1 - input.type(torch.int8)
        if self.order == 1:
            return (1 - u).type(self.stype)
        stage = 1 - self.scalers[-1](u)
        for k in range(self.order - 1, 0, -1):
            scaled = u if k == 1 else self.scalers[k - 2](u)
            nested = self.delays[k - 1](stage)
            product = self.multipliers[k - 1](scaled, nested)
            # No operation owns unipolar complementation, so the Horner NOT is direct.
            stage = 1 - product
        return stage.type(self.stype)

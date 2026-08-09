import torch

from napl.sim.base import napl_base
from loguru import logger


class add_any_dyn(napl_base):
    """
    Add spike streams with an output scale supplied at every timestep.

    Use this streaming accumulator when the carry scale is a runtime input
    rather than a construction constant, as when a controller retunes the
    reduction while the stream runs. It supports unipolar and bipolar
    rate-coded inputs.

    The target reductions for a call made with scale :math:`s` are

    .. math::

       p_y = \\frac{1}{s}\\sum_i p_i
       \\quad (\\text{unipolar}),\\qquad
       v_y = \\frac{1}{s}\\sum_i v_i
       \\quad (\\text{bipolar}).

    The output removes the reduced dimension. Every value in
    ``1 <= scale <= scale_max`` is supported, and a ``scale`` outside that range
    raises. The threshold, the subtracted carry, and the bipolar centering
    offset ``(entry - scale) / 2`` all follow the scale passed to the current
    call, so each output spike represents the scale in force at its fire time
    and a stream with a changing scale conserves mass against
    ``sum(scale at each fire)``.

    A finite accumulator width saturates the running sum, so the realized rate
    departs from the target once the reduced sum stays outside the
    representable range.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import add_any_dyn

        adder = add_any_dyn({'polarity': 'unipolar', 'scale_max': 4, 'width': 10})
        output = adder(torch.tensor([[1, 0], [1, 1]], dtype=torch.int8), 2, dim=0)

    .. container:: api-references

        .. rubric:: References

    """


    def __init__(
            self,
            config={
                'polarity' : 'bipolar',
                'scale_max' : 2,
                'width' : 10,
            }
        ):
        """
        Configure the largest supported carry scale and the accumulator width.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **scale_max**: Largest accumulated amount a call may request for one output spike; the default is ``2``.
              - **width**: Signed accumulator width in bits; the default is ``10``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'scale_max', 'width'], polarity_required=True)

        # For scale >= entry the static bound
        # 2 ** (width - 1) - 1 >= (scale - grid) + delta_max keeps the accumulator clear of
        # saturation for every input, where delta_max is the largest per-timestep step
        # (entry when unipolar, (entry + scale) / 2 when bipolar) and grid is the accumulator
        # step, 0.5 for a bipolar half-integer accumulator (odd entry - scale) and 1 otherwise.
        # Because scale arrives at runtime, no static width argument covers every legal call.
        #: Signed accumulator width in bits.
        self.width = config['width']
        if not isinstance(self.width, int):
            message = f'add_any_dyn accumulator width must be int: got <{self.width}>.'
            logger.error(message)
            raise AssertionError(message)
        #: Largest value retained by the signed accumulator.
        self.acc_max = 2**(self.width-1) - 1
        #: Smallest value retained by the signed accumulator.
        self.acc_min = -2**(self.width-1)

        #: Largest accumulated amount a call may request for one output spike.
        self.scale_max = config['scale_max']
        if not isinstance(self.scale_max, int):
            message = f'add_any_dyn scale_max must be int: got <{self.scale_max}>.'
            logger.error(message)
            raise AssertionError(message)
        if self.scale_max > self.acc_max:
            message = (
                f'add_any_dyn scale_max <{self.scale_max}> exceeds accumulator maximum '
                f'<{self.acc_max}> for width <{self.width}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Running centered input sum used to decide when to emit a spike.
        self.accumulator: torch.Tensor
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
        #: Hardware latency and timing metadata for the combinational adder.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Clear the local accumulator.
        """
        self.accumulator.resize_(1).zero_()


    def forward(self, input, scale, entry=None, dim=-1):
        """
        Accumulate and emit one output timestep at the requested scale.

        Args:
            input: Current spike tensor, or a pre-reduced partial sum when
                ``dim=None``.
            scale: Accumulated amount required for one output spike on this
                call, an int scalar in ``1`` to ``scale_max``. A value outside
                that range raises ``AssertionError``.
            entry: Number of source streams used for the bipolar offset. It is
                inferred from ``input.size(dim)`` unless ``dim=None``.
            dim: Dimension to reduce. Set it to ``None`` for pre-reduced input;
                the default is ``-1``.

        Returns:
            A spike tensor with the reduced dimension removed. The call updates
            the accumulator and advances the module by one timestep.

        **Example:**

        .. code-block:: python

            output = adder(torch.tensor([1, 1], dtype=torch.int8), 2, dim=0)
        """
        # bool is a subclass of int, so isinstance would admit True/False here.
        if type(scale) is not int:
            message = f'add_any_dyn scale must be an int scalar: got <{scale}>.'
            logger.error(message)
            raise AssertionError(message)
        if scale < 1 or scale > self.scale_max:
            message = (
                f'add_any_dyn scale <{scale}> outside the supported range <1> to '
                f'scale_max <{self.scale_max}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        offset = 0
        if self.polarity == 'bipolar':
            if entry is None:
                if dim is None:
                    message = 'add_any_dyn with pre-reduced input (dim=None) requires an explicit <entry>.'
                    logger.error(message)
                    raise AssertionError(message)
                entry = input.size()[dim]
            offset = (entry - scale)/2

        if dim is None:
            acc_delta = input.type(self.ntype) - offset
        else:
            acc_delta = torch.sum(input, dim, dtype=self.ntype)
            acc_delta.sub_(offset)
        # For scale < entry no static width is sufficient: at most one spike is emitted per
        # timestep, so the accumulator drains by at most scale per step while the inflow can
        # exceed that. Correctness is then conditional on the long-run mean inflow staying
        # below scale, width supplies burst headroom, and the clamp below is the backstop.
        # Since scale arrives at runtime the clamp is reachable by contract here and is the
        # only guard against a diverging accumulator.
        # The scalar initial state broadcasts out of place; matching shapes update in place.
        if self.accumulator.shape == acc_delta.shape:
            self.accumulator.add_(acc_delta).clamp_(self.acc_min, self.acc_max)
        else:
            updated = self.accumulator.add(acc_delta).clamp(self.acc_min, self.acc_max)
            self.accumulator.resize_as_(updated).copy_(updated.detach())
        output = torch.ge(self.accumulator, scale).type(self.ntype)
        # With scale > 0, emitting a carry preserves the accumulator bounds.
        self.accumulator.sub_(output, alpha=scale)
        return output.type(self.stype)

import torch
import math

from napl.sim.base import napl_base
from napl.utils import pow2_lshift
from loguru import logger


class add_scale_dyn(napl_base):
    """
    Add spike streams with a fixed-point output scale supplied at every timestep.

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

    The output removes the reduced dimension. Every positive finite ``scale`` up
    to ``scale_max`` is supported: each call quantizes its ``scale`` to the
    nearest multiple of ``2 ** -fracwidth``, with a tie going to the even
    multiple, and a request rounding to zero raw units or rounding above
    ``scale_max`` raises. The quantized value is reported as :attr:`scale`. The threshold, the
    subtracted carry, and the bipolar centering offset ``(entry - scale) / 2``
    all follow the quantized scale of the current call, so each output spike
    represents the scale in force at its fire time and a stream with a changing
    scale conserves mass against ``sum(quantized scale at each fire)``.

    The hardware counterpart takes ``scale`` on the integer port ``i_scale`` and
    exists only at ``fracwidth`` 0, so a caller whose calls must match the RTL
    passes an integer ``scale``. This class accepts any positive float up to
    ``scale_max`` and quantizes it, and no check here rejects a non-integer
    request.

    A finite accumulator saturates the running sum, so the realized rate departs
    from the target once the reduced sum stays outside the representable range.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import add_scale_dyn

        adder = add_scale_dyn({'polarity': 'unipolar', 'scale_max': 4, 'intwidth': 10, 'fracwidth': 4})
        output = adder(torch.tensor([[1, 0], [1, 1]], dtype=torch.int8), 1.5, dim=0)

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """


    def __init__(
            self,
            config={
                'polarity' : 'bipolar',
                'scale_max' : 2,
                'intwidth' : 10,
                'fracwidth' : 0,
            }
        ):
        """
        Configure the largest supported carry scale and the fixed-point accumulator format.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **scale_max**: Largest accumulated amount a call may request for one output spike, a positive finite number quantized to the nearest multiple of ``2 ** -fracwidth`` with ties going to the even multiple; the default is ``2``.
              - **intwidth**: Integer bits of the signed accumulator, including the sign bit; the default is ``10``.
              - **fracwidth**: Fractional bits of the signed accumulator; the default is ``0``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'scale_max', 'intwidth', 'fracwidth'], polarity_required=True)

        #: Integer bits of the signed accumulator, including the sign bit.
        self.intwidth = config['intwidth']
        if type(self.intwidth) is not int or self.intwidth < 1:
            message = f'Invalid intwidth: <{self.intwidth}>; legal values: an integer of at least 1.'
            logger.error(message)
            raise AssertionError(message)
        #: Fractional bits of the signed accumulator.
        self.fracwidth = config['fracwidth']
        if type(self.fracwidth) is not int or self.fracwidth < 0:
            message = f'Invalid fracwidth: <{self.fracwidth}>; legal values: a non-negative integer.'
            logger.error(message)
            raise AssertionError(message)

        #: Smallest effective carry scale, ``2 ** -fracwidth``.
        self.grid = 2.0 ** (-self.fracwidth)
        # The accumulator is carried in raw units of 2 ** -fracwidth, dropping to half raw units when a bipolar offset is odd, so its hardware equivalent is a signed integer register of intwidth + fracwidth + 1 bits.
        #: Largest value retained by the signed accumulator, in raw units.
        self.acc_max = 2**(self.intwidth+self.fracwidth-1) - 1
        #: Smallest value retained by the signed accumulator, in raw units.
        self.acc_min = -2**(self.intwidth+self.fracwidth-1)

        requested = config['scale_max']
        # round() raises on a non-finite value, so the finiteness test runs before the quantization
        # below, where round() also resolves a scale sitting exactly between two grid points to the
        # even multiple.
        if type(requested) not in (int, float) or requested <= 0 or not math.isfinite(requested):
            message = f'Invalid scale_max: <{requested}>; legal values: a positive finite int or float.'
            logger.error(message)
            raise AssertionError(message)
        #: Largest requestable carry scale in accumulator raw units.
        self.scale_max_raw = round(float(requested) * 2**self.fracwidth)
        if self.scale_max_raw == 0:
            message = f'Invalid scale_max: <{requested}>; legal values: greater than half of <{self.grid}>.'
            logger.error(message)
            raise AssertionError(message)
        #: Largest accumulated amount a call may request for one output spike, the request quantized to the grid.
        self.scale_max = self.scale_max_raw * self.grid
        # The guard below rejects only scale_max_raw > acc_max, since no static accumulator format keeps a runtime scale clear of saturation for every legal call, so a config passing the guard runs with the accumulator clamping.
        if self.scale_max_raw > self.acc_max:
            message = (
                f'add_scale_dyn scale_max <{self.scale_max}> exceeds accumulator maximum '
                f'<{self.acc_max * self.grid}> for intwidth <{self.intwidth}> and '
                f'fracwidth <{self.fracwidth}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Carry scale of the most recent call, the request quantized to the grid, or ``None`` before the first call.
        self.scale = None
        #: Running centered input sum in raw units, used to decide when to emit a spike.
        self.accumulator: torch.Tensor
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
        #: Hardware latency and timing metadata for the combinational adder.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Clear the local accumulator and the reported carry scale.

        The scale returns to ``None`` because no call has selected one yet.
        """
        self.accumulator.resize_(1).zero_()
        self.scale = None


    def forward(self, input, scale, entry=None, dim=-1):
        """
        Accumulate and emit one output timestep at the requested scale.

        Args:
            input: Current spike tensor, or a pre-reduced partial sum when
                ``dim=None``.
            scale: Accumulated amount required for one output spike on this
                call, a positive finite scalar quantized to the nearest multiple
                of ``2 ** -fracwidth`` with ties going to the even multiple. A
                request rounding to zero raw units or rounding above ``scale_max`` raises
                ``AssertionError``.
            entry: Number of source streams used for the bipolar offset. It is
                inferred from ``input.size(dim)`` unless ``dim=None``.
            dim: Dimension to reduce. Set it to ``None`` for pre-reduced input;
                the default is ``-1``.

        Returns:
            A spike tensor with the reduced dimension removed. The call updates
            the accumulator, reports the quantized scale as :attr:`scale`, and
            advances the module by one timestep.

        **Example:**

        .. code-block:: python

            output = adder(torch.tensor([1, 1], dtype=torch.int8), 1.5, dim=0)
        """
        # bool is a subclass of int, so isinstance would admit True/False here.
        if type(scale) not in (int, float):
            message = f'add_scale_dyn scale must be an int or float scalar: got <{scale}>.'
            logger.error(message)
            raise AssertionError(message)
        # round() raises on a non-finite value, so the finiteness test runs before the quantization
        # below, where round() also resolves a scale sitting exactly between two grid points to the
        # even multiple.
        if scale <= 0 or not math.isfinite(scale):
            message = f'Invalid scale: <{scale}>; legal values: a positive finite int or float.'
            logger.error(message)
            raise AssertionError(message)
        scale_raw = round(float(scale) * 2**self.fracwidth)
        if scale_raw == 0:
            message = f'Invalid scale: <{scale}>; legal values: greater than half of <{self.grid}>.'
            logger.error(message)
            raise AssertionError(message)
        # The comparison uses the quantized request, so a scale rounding down onto scale_max is
        # accepted and the accumulator format still covers it.
        if scale_raw > self.scale_max_raw:
            message = (
                f'add_scale_dyn scale <{scale}> exceeds scale_max <{self.scale_max}>.'
            )
            logger.error(message)
            raise AssertionError(message)
        self.scale = scale_raw * self.grid

        offset = 0.0
        if self.polarity == 'bipolar':
            if entry is None:
                if dim is None:
                    message = 'add_scale_dyn with pre-reduced input (dim=None) requires an explicit <entry>.'
                    logger.error(message)
                    raise AssertionError(message)
                entry = input.size()[dim]
            offset = (entry * 2**self.fracwidth - scale_raw)/2

        # One input spike is one unit of value, which is 2 ** fracwidth raw accumulator units;
        # torch shifts are integer-only, so the shim scales the float spike tensor.
        if dim is None:
            acc_delta = pow2_lshift(input.type(self.ntype), self.fracwidth)
        else:
            acc_delta = pow2_lshift(torch.sum(input, dim, dtype=self.ntype), self.fracwidth)
        acc_delta.sub_(offset)
        # The accumulator drains by at most scale per timestep while the inflow can exceed that,
        # so the clamp below is the only guard against a diverging accumulator.
        # The scalar initial state broadcasts out of place; matching shapes update in place.
        if self.accumulator.shape == acc_delta.shape:
            self.accumulator.add_(acc_delta).clamp_(self.acc_min, self.acc_max)
        else:
            updated = self.accumulator.add(acc_delta).clamp(self.acc_min, self.acc_max)
            self.accumulator.resize_as_(updated).copy_(updated.detach())
        output = torch.ge(self.accumulator, scale_raw).type(self.ntype)
        # With scale > 0, emitting a carry preserves the accumulator bounds.
        self.accumulator.sub_(output, alpha=scale_raw)
        return output.type(self.stype)

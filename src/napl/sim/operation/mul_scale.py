import torch
import math

from napl.sim.base import napl_base
from loguru import logger


class mul_scale(napl_base):
    """
    Multiply a spike stream by a fixed-point scale.

    Use this streaming accumulator when a single rate-coded stream must be
    multiplied by a constant that need not be an integer. It supports unipolar
    and bipolar rate-coded inputs and takes one input stream, so it performs no
    reduction and returns an output of the same shape as its input.

    The target products are

    .. math::

       p_y = p_\\mathit{in}\\,\\mathit{scale}
       \\quad (\\text{unipolar}),\\qquad
       v_y = v_\\mathit{in}\\,\\mathit{scale}
       \\quad (\\text{bipolar}).

    This is a *derived* operation: the fixed-point-scale dual of
    :class:`~napl.sim.operation.div_scale`, which divides by the same grid.

    ``scale`` is any positive finite number, quantized on construction to the
    nearest multiple of ``2 ** -fracwidth``, with a tie going to the even
    multiple. The quantized value is reported as ``scale`` and drives the
    product; a request rounding to zero raw units raises. A finite accumulator
    saturates the running sum, so the realized rate departs from the target
    once the accumulated input stays outside the representable range. For
    ``scale > 1`` the product amplifies: each input spike adds ``scale`` units
    while the accumulator drains by only one unit per fired spike, so once
    ``p_in * scale`` exceeds 1 the accumulator rails and the output saturates at
    rate 1. The caller narrows the input so ``x * scale`` stays a legal rate.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import mul_scale

        scaler = mul_scale({'polarity': 'unipolar', 'scale': 2, 'intwidth': 8, 'fracwidth': 4})
        output = scaler(torch.tensor([1, 0], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """


    def __init__(
            self,
            config={
                'polarity' : 'bipolar',
                'scale' : 2,
                'intwidth' : 8,
                'fracwidth' : 4,
            }
        ):
        """
        Configure the product scale and the fixed-point accumulator format.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **scale**: Factor applied to the input rate, a positive finite number quantized to the nearest multiple of ``2 ** -fracwidth`` with ties going to the even multiple; the default is ``2``.
              - **intwidth**: Integer bits of the signed accumulator, including the sign bit; the default is ``8``.
              - **fracwidth**: Fractional bits of the signed accumulator; the default is ``4``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'scale', 'intwidth', 'fracwidth'], polarity_required=True)

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

        #: Smallest value the fixed-point grid resolves, ``2 ** -fracwidth``.
        self.grid = 2.0 ** (-self.fracwidth)
        # The accumulator is carried in raw units of 2 ** -fracwidth, dropping to half raw units when a bipolar offset is odd, so its hardware equivalent is a signed integer register of intwidth + fracwidth + 1 bits.
        #: One value unit in accumulator raw units, ``2 ** fracwidth``; the fire threshold and the amount drained per fired spike.
        self.unit = 2**self.fracwidth
        #: Largest value retained by the signed accumulator, in raw units.
        self.acc_max = 2**(self.intwidth+self.fracwidth-1) - 1
        #: Smallest value retained by the signed accumulator, in raw units.
        self.acc_min = -2**(self.intwidth+self.fracwidth-1)

        requested = config['scale']
        # round() raises on a non-finite value, so the finiteness test runs before the quantization
        # below, where round() also resolves a scale sitting exactly between two grid points to the
        # even multiple.
        if type(requested) not in (int, float) or requested <= 0 or not math.isfinite(requested):
            message = f'Invalid scale: <{requested}>; legal values: a positive finite int or float.'
            logger.error(message)
            raise AssertionError(message)
        #: Factor applied to the input rate, in accumulator raw units.
        self.scale_raw = round(float(requested) * 2**self.fracwidth)
        if self.scale_raw == 0:
            message = f'Invalid scale: <{requested}>; legal values: greater than half of <{self.grid}>.'
            logger.error(message)
            raise AssertionError(message)
        #: Effective factor applied to the input rate, the request quantized to the grid.
        self.scale = self.scale_raw * self.grid
        # Each input spike adds scale_raw raw units, so the delta itself must fit the accumulator. The
        # guard rejects only scale_raw > acc_max; keeping the accumulator clear of saturation is a
        # stronger sizing rule left to the caller, so a config that passes the guard without meeting it
        # still runs with the accumulator clamping.
        if self.scale_raw > self.acc_max:
            message = (
                f'mul_scale scale <{self.scale}> exceeds accumulator maximum '
                f'<{self.acc_max * self.grid}> for intwidth <{self.intwidth}> and '
                f'fracwidth <{self.fracwidth}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        # The bipolar centering offset is the negation of div_scale's: draining one unit per fired
        # spike while adding scale_raw per input spike, this offset makes the mean output rate settle
        # at v_out = v_in * scale (derivation in forward()).
        #: Bipolar centering offset in raw units, zero when unipolar.
        self.offset = 0.0
        if self.polarity == 'bipolar':
            self.offset = (self.scale_raw - 2**self.fracwidth)/2
        #: Running centered input sum in raw units, used to decide when to emit a spike.
        self.accumulator: torch.Tensor
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
        #: Hardware latency and timing metadata for the combinational scaler.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Clear the local accumulator.
        """
        self.accumulator.resize_(1).zero_()


    def forward(self, input):
        """
        Accumulate and emit one output timestep.

        Args:
            input: Current spike tensor of any shape.

        Returns:
            A spike tensor shaped like ``input``. The call updates the
            accumulator and advances the module by one timestep.

        **Example:**

        .. code-block:: python

            output = scaler(torch.tensor([1, 0], dtype=torch.int8))
        """
        # One input spike carries scale worth of value, which is scale_raw raw accumulator units;
        # scale_raw need not be a power of two, so this is a plain multiply, not a bit shift. In
        # bipolar, subtracting the offset recenters the sum so the mean output rate settles at
        # p_out = p_in * scale_raw / unit - offset / unit = 1/2 + v_in * scale / 2, i.e. v_out = v_in * scale.
        acc_delta = input.type(self.ntype) * self.scale_raw
        acc_delta.sub_(self.offset)
        # Matching shapes update in place and any shape mismatch broadcasts out of place, with the
        # clamp holding the accumulator bounded when scale > 1 lets inflow outpace the one-unit drain.
        if self.accumulator.shape == acc_delta.shape:
            self.accumulator.add_(acc_delta).clamp_(self.acc_min, self.acc_max)
        else:
            updated = self.accumulator.add(acc_delta).clamp(self.acc_min, self.acc_max)
            self.accumulator.resize_as_(updated).copy_(updated.detach())
        # One fired spike per timestep at most, so the emitted stream is always a legal 0/1 rate; the
        # accumulator and clamp keep the running sum bounded when the target rate would exceed 1.
        output = torch.ge(self.accumulator, self.unit).type(self.ntype)
        self.accumulator.sub_(output, alpha=self.unit)
        return output.type(self.stype)

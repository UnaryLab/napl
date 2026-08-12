import torch
import math

from napl.sim.base import napl_base
from napl.utils import pow2_lshift
from loguru import logger


class add_scale(napl_base):
    """
    Add spike streams with a configurable fixed-point output scale.

    Use this streaming accumulator when the reduced sum must emit one output
    spike per ``scale`` accumulated units and that scale need not be an integer.
    It supports unipolar and bipolar rate-coded inputs.

    The target reductions are

    .. math::

       p_y = \\frac{1}{\\mathit{scale}}\\sum_i p_i
       \\quad (\\text{unipolar}),\\qquad
       v_y = \\frac{1}{\\mathit{scale}}\\sum_i v_i
       \\quad (\\text{bipolar}).

    ``scale`` is any positive finite number, quantized on construction to the
    nearest multiple of ``2 ** -fracwidth``, with a tie going to the even
    multiple. The quantized value is reported as ``scale`` and drives the
    reduction; a request rounding to zero raw units raises. The output removes
    the reduced dimension. A finite accumulator saturates the running sum, so
    the realized rate departs from the target once the reduced sum stays outside
    the representable range.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import add_scale

        adder = add_scale({'polarity': 'unipolar', 'scale': 1.5, 'intwidth': 10, 'fracwidth': 4})
        output = adder(torch.tensor([[1, 0], [1, 1]], dtype=torch.int8), dim=0)

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """


    def __init__(
            self,
            config={
                'polarity' : 'bipolar',
                'scale' : 2,
                'intwidth' : 10,
                'fracwidth' : 0,
            }
        ):
        """
        Configure the carry scale and the fixed-point accumulator format.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **scale**: Accumulated amount required for one output spike, a positive finite number quantized to the nearest multiple of ``2 ** -fracwidth`` with ties going to the even multiple; the default is ``2``.
              - **intwidth**: Integer bits of the signed accumulator, including the sign bit; the default is ``10``.
              - **fracwidth**: Fractional bits of the signed accumulator; the default is ``0``.
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
        #: Carry threshold in accumulator raw units.
        self.scale_raw = round(float(requested) * 2**self.fracwidth)
        if self.scale_raw == 0:
            message = f'Invalid scale: <{requested}>; legal values: greater than half of <{self.grid}>.'
            logger.error(message)
            raise AssertionError(message)
        #: Accumulated amount consumed when an output spike is emitted, the request quantized to the grid.
        self.scale = self.scale_raw * self.grid
        # The guard below rejects only scale_raw > acc_max; keeping the accumulator clear of saturation is a stronger sizing rule left to the caller, so a config that passes the guard without meeting it still runs with the accumulator clamping.
        if self.scale_raw > self.acc_max:
            message = (
                f'add_scale scale <{self.scale}> exceeds accumulator maximum '
                f'<{self.acc_max * self.grid}> for intwidth <{self.intwidth}> and '
                f'fracwidth <{self.fracwidth}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Bipolar centering offset in raw units, inferred from the input count on first use.
        self.offset = 0.0
        #: Running centered input sum in raw units, used to decide when to emit a spike.
        self.accumulator: torch.Tensor
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
        #: Whether the next call must infer input-dependent state.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the combinational adder.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Clear the local accumulator and arm the next call as the first, so the
        bipolar offset is inferred again.
        """
        self.accumulator.resize_(1).zero_()
        self.is_first_call = True


    def forward(self, input, entry=None, dim=-1):
        """
        Accumulate and emit one output timestep.

        Args:
            input: Current spike tensor, or a pre-reduced partial sum when
                ``dim=None``.
            entry: Number of source streams used for the bipolar offset. Only
                the first call after ``reset()`` reads it, and it is inferred
                from ``input.size(dim)`` unless ``dim=None``.
            dim: Dimension to reduce. Set it to ``None`` for pre-reduced input;
                the default is ``-1``.

        Returns:
            A spike tensor with the reduced dimension removed. The call updates
            the accumulator and advances the module by one timestep.

        **Example:**

        .. code-block:: python

            output = adder(torch.tensor([1, 1], dtype=torch.int8), dim=0)
        """
        if self.is_first_call:
            if self.polarity == 'bipolar':
                if entry is None:
                    if dim is None:
                        message = 'add_scale with pre-reduced input (dim=None) requires an explicit <entry>.'
                        logger.error(message)
                        raise AssertionError(message)
                    entry = input.size()[dim]
                self.offset = (entry * 2**self.fracwidth - self.scale_raw)/2

            self.is_first_call = False

        # One input spike is one unit of value, which is 2 ** fracwidth raw accumulator units;
        # torch shifts are integer-only, so the shim scales the float spike tensor.
        if dim is None:
            acc_delta = pow2_lshift(input.type(self.ntype), self.fracwidth)
        else:
            acc_delta = pow2_lshift(torch.sum(input, dim, dtype=self.ntype), self.fracwidth)
        acc_delta.sub_(self.offset)
        # Matching shapes update in place and any shape mismatch broadcasts out of place, with the
        # clamp holding the accumulator bounded when scale < entry lets inflow outpace the drain.
        if self.accumulator.shape == acc_delta.shape:
            self.accumulator.add_(acc_delta).clamp_(self.acc_min, self.acc_max)
        else:
            updated = self.accumulator.add(acc_delta).clamp(self.acc_min, self.acc_max)
            self.accumulator.resize_as_(updated).copy_(updated.detach())
        output = torch.ge(self.accumulator, self.scale_raw).type(self.ntype)
        # With scale > 0, emitting a carry preserves the accumulator bounds.
        self.accumulator.sub_(output, alpha=self.scale_raw)
        return output.type(self.stype)

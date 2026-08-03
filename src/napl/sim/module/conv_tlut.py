import torch
import math

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger
from napl.sim.module._shared import (
    _init_conv_params,
    _conv2d_binary,
    _linear_tlut_fxpfxp_fn,
    _linear_tlut_fxpfp_fn,
    _linear_tlut_fpfp_fn,
    _TLUT_FP_WIDTH,
)


class conv_tlut(napl_base):
    """Apply a temporal-LUT approximation of ``torch.nn.Conv2d``.

    Use this single-shot trainable layer to decompose either convolution inputs or
    weights into temporal digits. It supports modes
    fxpfxp/fxpfp/fpfp via the (formati, formatw) pair. Single-shot; trains via STE.
    Approximates nn.Conv2d within the temporal-decomposition bound.
    ``fxpfxp``, ``fxpfp``, and ``fpfp`` with ``groups=1`` and zero padding.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import conv_tlut

        layer = conv_tlut(1, 2, 3, padding=1,
                          config={"temporal": "i", "widtht": 4,
                                  "formati": "fxp", "widthi": 8,
                                  "formatw": "fxp", "widthw": 8})
        output = layer(torch.zeros(1, 1, 4, 4))
    """
    streaming = False


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bias=True, weight_ext=None, bias_ext=None,
                 config={'temporal': 'i', 'widtht': 4, 'formati': 'fxp', 'widthi': 8, 'quantilei': 1,
                         'formatw': 'fxp', 'widthw': 8, 'quantilew': 1, 'cycle': None, 'rounding': 'round'}):
        """Configure convolution geometry and temporal decomposition.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size accepted as an integer or pair.
            stride: Convolution stride. Defaults to ``1``.
            padding: Symmetric zero padding. Defaults to ``0``.
            dilation: Kernel dilation. Defaults to ``1``.
            bias: Create a trainable bias when ``True``. Defaults to ``True``.
            weight_ext: Optional initial convolution weight. Defaults to ``None``.
            bias_ext: Optional initial bias. Defaults to ``None``.
            config: Configuration mapping with these keys:

                * **temporal** - ``"i"``/``"input"`` or ``"w"``/``"weight"``.
                  Defaults to ``"i"``.
                * **widtht** - Bits per temporal digit. Defaults to ``4``.
                * **formati**, **formatw** - ``"fxp"`` or a supported floating
                  format: ``"bfloat16"``, ``"float16"``, or ``"float32"``.
                  Both default to ``"fxp"``.
                * **widthi**, **widthw** - Fixed-point widths. Both default to ``8``.
                * **quantilei**, **quantilew** - Scaling quantiles. Both default
                  to ``1``.
                * **cycle** - Active cycles, capped at ``2 ** widtht``. Defaults
                  to ``None``, which selects the cap.
                * **rounding** - Fixed-point rounding mode. Defaults to ``"round"``.
                * **name** - Optional instance label. Defaults to ``None``.
        """
        super().__init__(config, [])
        #: Spatial height and width of the convolution kernel.
        self.kernel_size = kernel_size
        #: Spatial step between adjacent convolution windows.
        self.stride = stride
        #: Symmetric zero padding applied to the input.
        self.padding = padding
        #: Spacing between kernel elements.
        self.dilation = dilation
        #: Operand decomposed into temporal digits.
        self.temporal = config.get('temporal', 'i').lower()
        #: Number of bits represented by each temporal digit.
        self.widtht = config.get('widtht', 4)
        #: Numeric format used for input values.
        self.formati = config.get('formati', 'fxp').lower()
        #: Numeric format used for weight values.
        self.formatw = config.get('formatw', 'fxp').lower()
        #: Fixed-point width used for input values.
        self.widthi = config.get('widthi', 8)
        #: Fixed-point width used for weight values.
        self.widthw = config.get('widthw', 8)
        #: Input-magnitude quantile used to choose the scaling shift.
        self.quantilei = config.get('quantilei', 1)
        #: Weight-magnitude quantile used to choose the scaling shift.
        self.quantilew = config.get('quantilew', 1)
        #: Rounding mode used during fixed-point conversion.
        self.rounding = config.get('rounding', 'round').lower()
        assert self.temporal in ('i', 'input', 'w', 'weight'), \
            logger.error(f"conv_tlut 'temporal' must be one of ['i','input','w','weight'], got {self.temporal}.")

        if self.formati == 'fxp' and self.formatw == 'fxp':
            #: Arithmetic path selected from the input and weight formats.
            self.mode = 'fxpfxp'
        elif self.formati != 'fxp' and self.formatw != 'fxp':
            #: Arithmetic path selected from the input and weight formats.
            self.mode = 'fpfp'
        else:
            #: Arithmetic path selected from the input and weight formats.
            self.mode = 'fxpfp'

        #: Maximum number of temporal cycles per product.
        self.cycle_max = 2 ** self.widtht
        cycle_cfg = config.get('cycle', None)
        #: Number of temporal cycles used per product.
        self.cycle_act = self.cycle_max if cycle_cfg is None else min(cycle_cfg, self.cycle_max)
        #: Input magnitude width excluding its sign bit.
        self.widthi_mag = self.widthi - 1
        #: Weight magnitude width excluding its sign bit.
        self.widthw_mag = self.widthw - 1
        if self.temporal in ('i', 'input'):
            fmt = self.formati
            #: Bit width of the operand decomposed into temporal digits.
            self.width = self.widthi - 1 if fmt == 'fxp' else _TLUT_FP_WIDTH[fmt]
        else:
            fmt = self.formatw
            #: Bit width of the operand decomposed into temporal digits.
            self.width = self.widthw - 1 if fmt == 'fxp' else _TLUT_FP_WIDTH[fmt]
        #: Number of temporal digits required for the selected operand.
        self.degree = int(math.ceil(self.width / self.widtht))
        #: Leading padding bits in the temporal decomposition.
        self.delta = int(self.degree * self.widtht - self.width)
        _init_conv_params(self, in_channels, out_channels, kernel_size, bias, weight_ext, bias_ext)


    def _reset(self):
        """Reset local execution state.

        This single-shot layer has no mutable run state. Trainable parameters are
        unchanged.
        """
        pass


    def forward(self, input):
        """Apply temporal-LUT convolution.

        Args:
            input: Numeric NCHW tensor shaped
                ``(batch, in_channels, height, width)``.

        Returns:
            Numeric NCHW tensor with ``out_channels`` and the configured output
            geometry.

        The call does not change persistent state or ``timestep_cur``. The
        selected custom autograd path supplies a straight-through gradient.
        """
        cp, cn = self.cycle_act, -self.cycle_act
        if self.mode == 'fxpfxp':
            fn = lambda i2d, w2d: _linear_tlut_fxpfxp_fn.apply(i2d, w2d, None, self.temporal, self.widthi_mag,
                self.widthw_mag, self.widtht, self.degree, self.delta, cp, cn, self.rounding, self.quantilei, self.quantilew)
        elif self.mode == 'fxpfp':
            fn = lambda i2d, w2d: _linear_tlut_fxpfp_fn.apply(i2d, w2d, None, self.temporal, self.width,
                self.widtht, self.degree, self.delta, cp, cn, self.rounding, self.quantilei, self.quantilew)
        else:
            fn = lambda i2d, w2d: _linear_tlut_fpfp_fn.apply(i2d, w2d, None, self.temporal, self.width,
                self.widtht, self.degree, self.delta, cp, cn)
        return _conv2d_binary(input, self.weight, self.bias, self.kernel_size, self.stride,
                              self.padding, self.dilation, fn)

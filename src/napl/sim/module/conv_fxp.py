import torch
import math

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger
from napl.sim.module._shared import _init_conv_params, _conv2d_binary, _linear_fxp_fn


class conv_fxp(napl_base):
    """Apply a trainable fixed-point approximation of ``torch.nn.Conv2d``.

    Use this single-shot layer for quantization-aware convolution with
    ``groups=1`` and zero padding. It lowers convolution to image columns, applies
    the fixed-point linear kernel, and folds the result back to NCHW.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import conv_fxp

        layer = conv_fxp(1, 2, 3, padding=1)
        output = layer(torch.zeros(1, 1, 4, 4))
    """
    streaming = False


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bias=True, weight_ext=None, bias_ext=None,
                 config={'widthi': 8, 'quantilei': 1, 'widthw': 8, 'quantilew': 1, 'rounding': 'round'}):
        """Configure the convolution geometry and fixed-point approximation.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size accepted as an integer or pair.
            stride: Convolution stride. Defaults to ``1``.
            padding: Symmetric zero padding. Defaults to ``0``.
            dilation: Kernel dilation. Defaults to ``1``.
            bias: Create a trainable bias when ``True``. Defaults to ``True``.
            weight_ext: Optional initial weight shaped
                ``(out_channels, in_channels, kernel_height, kernel_width)``.
                Defaults to ``None``.
            bias_ext: Optional initial bias shaped ``(out_channels,)``. Defaults
                to ``None``.
            config: Configuration mapping with **widthi** and **widthw** (both
                default ``8``), **quantilei** and **quantilew** (both default
                ``1``), and **rounding** (default ``"round"``). **name** is an
                optional instance label and defaults to ``None``.
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
        #: Largest positive input magnitude represented by the quantized kernel.
        self.max_abs_i = 2 ** (self.widthi - 1)
        #: Largest positive weight magnitude represented by the quantized kernel.
        self.max_abs_w = 2 ** (self.widthw - 1)
        _init_conv_params(self, in_channels, out_channels, kernel_size, bias, weight_ext, bias_ext)


    def _reset(self):
        """Reset local execution state.

        This single-shot layer has no mutable run state. Trainable parameters are
        unchanged.
        """
        pass


    def forward(self, input):
        """Apply fixed-point convolution.

        Args:
            input: Numeric NCHW tensor shaped
                ``(batch, in_channels, height, width)``.

        Returns:
            Numeric NCHW tensor with ``out_channels`` and the configured output
            geometry.

        The call computes dynamic quantization shifts without changing persistent
        state or ``timestep_cur``. Backpropagation uses a straight-through linear
        gradient through the lowered convolution.
        """
        rshift_i, rshift_w, _ = rshift_offset(input, self.weight, self.widthi - 1, self.widthw - 1,
                                              self.rounding, self.quantilei, self.quantilew)
        rshift_o = 0 - rshift_i - rshift_w
        fn = lambda i2d, w2d: _linear_fxp_fn.apply(i2d, w2d, None, rshift_i, rshift_w, rshift_o,
                                                   self.max_abs_i, self.max_abs_w, True)
        return _conv2d_binary(input, self.weight, self.bias, self.kernel_size, self.stride,
                              self.padding, self.dilation, fn)

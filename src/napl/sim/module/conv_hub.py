import torch
import math

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger
from napl.sim.module._shared import _init_conv_params, _conv2d_binary, _build_hub_map, _linear_hub_fn


class conv_hub(napl_base):
    """Apply a hybrid unary-binary approximation of ``torch.nn.Conv2d``.

    Use this single-shot trainable layer when convolution products should use a
    unary multiplication lookup map while the interface remains numeric. It
    supports ``groups=1``, zero padding, and requires ``widthi == widthw``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import conv_hub

        layer = conv_hub(1, 2, 3, padding=1,
                         config={"widthi": 4, "widthw": 4, "cycle": 8})
        output = layer(torch.zeros(1, 1, 4, 4))
    """
    streaming = False


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bias=True, weight_ext=None, bias_ext=None,
                 config={'widthi': 8, 'rngi': 'sobol', 'quantilei': 1, 'widthw': 8, 'rngw': 'sobol',
                         'quantilew': 1, 'cycle': 128, 'rounding': 'round'}):
        """Configure the convolution geometry and HUB lookup map.

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
            config: Configuration mapping with equal **widthi** and **widthw**
                (both default ``8``), **rngi** and **rngw** (both default
                ``"sobol"``), **quantilei** and **quantilew** (both default
                ``1``), **cycle** (declared default ``128``; ``None`` selects the
                maximum), and **rounding** (default ``"round"``). **name** is an
                optional instance label and defaults to ``None``.

        **cycle** is capped at ``2 ** (widthi - 1)``. The generated lookup map is
        persistent non-trainable state.
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
        #: Quantization width used for input values.
        self.widthi = config.get('widthi', 8)
        #: Quantization width used for weight values.
        self.widthw = config.get('widthw', 8)
        assert self.widthi == self.widthw, \
            logger.error(f'conv_hub requires widthi == widthw (got {self.widthi}, {self.widthw}).')
        #: Number-sequence generator used for input values.
        self.rngi = config.get('rngi', 'sobol').lower()
        #: Number-sequence generator used for weight values.
        self.rngw = config.get('rngw', 'sobol').lower()
        #: Input-magnitude quantile used to choose the scaling shift.
        self.quantilei = config.get('quantilei', 1)
        #: Weight-magnitude quantile used to choose the scaling shift.
        self.quantilew = config.get('quantilew', 1)
        #: Rounding mode used during quantization.
        self.rounding = config.get('rounding', 'round').lower()
        #: Maximum cycle count supported by the unary product map.
        self.cycle_max, mapcbsg = _build_hub_map(self.widthi, self.widthw, self.rngi, self.rngw, self.ntype)
        cycle_cfg = config.get('cycle', None)
        #: Cycle count used for each unary product.
        self.cycle_act = self.cycle_max if cycle_cfg is None else min(cycle_cfg, self.cycle_max)
        #: Lookup map used to evaluate unary products.
        self.mapcbsg: torch.Tensor
        self.register_buffer('mapcbsg', mapcbsg)
        _init_conv_params(self, in_channels, out_channels, kernel_size, bias, weight_ext, bias_ext)


    def _reset(self):
        """Reset local execution state.

        This single-shot layer has no mutable run state. The lookup map and
        trainable parameters are unchanged.
        """
        pass


    def forward(self, input):
        """Apply HUB convolution.

        Args:
            input: Numeric NCHW tensor shaped
                ``(batch, in_channels, height, width)``.

        Returns:
            Numeric NCHW tensor with ``out_channels`` and the configured output
            geometry.

        The call reads the product lookup map and computes dynamic shifts without
        changing persistent state or ``timestep_cur``. Backpropagation uses a
        straight-through linear gradient.
        """
        rshift_i, rshift_w, rshift_o = rshift_offset(input, self.weight, self.widthi - 1, self.widthw - 1,
                                                     self.rounding, self.quantilei, self.quantilew)
        fn = lambda i2d, w2d: _linear_hub_fn.apply(i2d, w2d, None, rshift_i, rshift_w, rshift_o,
                                                   self.cycle_act, self.mapcbsg)
        return _conv2d_binary(input, self.weight, self.bias, self.kernel_size, self.stride,
                              self.padding, self.dilation, fn)

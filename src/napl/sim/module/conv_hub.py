import torch

from napl.utils import rshift_offset
from napl.sim.base import napl_base
from napl.sim.module._shared import _init_conv_params, _conv2d_binary, _build_hub_map, _linear_hub_fn
from loguru import logger

# Single source for every optional key: the signature default and the per-key fallback.
_DEFAULT_CONFIG = {
    'widthi': 8,
    'rngi': 'sobol',
    'quantilei': 1,
    'widthw': 8,
    'rngw': 'sobol',
    'quantilew': 1,
    'cycle': None,
    'rounding': 'round',
}


class conv_hub(napl_base):
    r"""Apply a hybrid unary-binary approximation of ``torch.nn.Conv2d``.

    Use this single-shot trainable layer when convolution products should use a
    unary multiplication lookup map while the interface remains numeric. It
    supports ``groups=1``, zero padding, and requires ``widthi == widthw``.

    The target is

    .. math::

       y = \mathrm{conv2d}(x, W) + b.

    Each product is replaced by the unary AND-count of the two quantized
    magnitudes, read from the lookup map :math:`M`,

    .. math::

       M_{a,b} \approx \frac{a\,b}{C},\qquad C = 2^{\text{width}-1},

    so the error is bounded by the unary-multiplication bound at **cycle**
    cycles.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import conv_hub

        layer = conv_hub(1, 2, 3, padding=1,
                         config={"widthi": 4, "widthw": 4, "cycle": 8})
        output = layer(torch.zeros(1, 1, 4, 4))

    .. container:: api-references

        .. rubric:: References

        *uSystolic: Byte-Crawling Unary Systolic Array*, HPCA, 2022.
    """
    streaming = False


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bias=True, weight_ext=None, bias_ext=None,
                 config=_DEFAULT_CONFIG):
        """Configure the convolution geometry and HUB lookup map.

        .. container:: api-parameter-list

            **Parameters:**

            - **in_channels** – Number of input channels.
            - **out_channels** – Number of output channels.
            - **kernel_size** – Kernel size accepted as an integer or pair.
            - **stride** – Convolution stride; the default is ``1``.
            - **padding** – Symmetric zero padding; the default is ``0``.
            - **dilation** – Kernel dilation; the default is ``1``.
            - **bias** – Create a trainable bias when ``True``; the default is ``True``.
            - **weight_ext** – Optional initial convolution weight; the default is ``None``.
            - **bias_ext** – Optional initial bias; the default is ``None``.
            - **config** – Configuration mapping. Omitted keys fall back to the same defaults.

              - **widthi**: Quantization width for input values, which must equal **widthw**; the default is ``8``.
              - **rngi**: Number-sequence generator for input values, one of ``"sobol"``, ``"rc"``, ``"tc"``; the default is ``"sobol"``.
              - **quantilei**: Input-magnitude scaling quantile; the default is ``1``.
              - **widthw**: Quantization width for weight values; the default is ``8``.
              - **rngw**: Number-sequence generator for weight values, one of ``"sobol"``, ``"rc"``, ``"tc"``; the default is ``"sobol"``.
              - **quantilew**: Weight-magnitude scaling quantile; the default is ``1``.
              - **cycle**: Active unary cycles, capped at ``2 ** (widthi - 1)``, where ``None`` selects the cap; the default is ``None``.
              - **rounding**: Dynamic-scaling rounding mode; the default is ``"round"``.
              - **name**: Optional instance label.

        The generated lookup map is persistent non-trainable state.
        """
        super().__init__(config, [], optional_key_list=list(_DEFAULT_CONFIG))
        cfg = {**_DEFAULT_CONFIG, **config}
        #: Spatial height and width of the convolution kernel.
        self.kernel_size = kernel_size
        #: Spatial step between adjacent convolution windows.
        self.stride = stride
        #: Symmetric zero padding applied to the input.
        self.padding = padding
        #: Spacing between kernel elements.
        self.dilation = dilation
        #: Quantization width used for input values.
        self.widthi = cfg['widthi']
        #: Quantization width used for weight values.
        self.widthw = cfg['widthw']
        if self.widthi != self.widthw:
            message = f'conv_hub requires widthi == widthw (got {self.widthi}, {self.widthw}).'
            logger.error(message)
            raise AssertionError(message)
        #: Number-sequence generator used for input values.
        self.rngi = cfg['rngi'].lower()
        #: Number-sequence generator used for weight values.
        self.rngw = cfg['rngw'].lower()
        #: Input-magnitude quantile used to choose the scaling shift.
        self.quantilei = cfg['quantilei']
        #: Weight-magnitude quantile used to choose the scaling shift.
        self.quantilew = cfg['quantilew']
        #: Rounding mode applied to the log2 magnitude that sets the scaling shift.
        self.rounding = cfg['rounding'].lower()
        #: Maximum cycle count supported by the unary product map.
        self.cycle_max, mapcbsg = _build_hub_map(self.widthi, self.widthw, self.rngi, self.rngw, self.ntype)
        cycle_cfg = cfg['cycle']
        #: Cycle count used for each unary product.
        self.cycle_act = self.cycle_max if cycle_cfg is None else min(cycle_cfg, self.cycle_max)
        #: Lookup map used to evaluate unary products.
        self.mapcbsg: torch.Tensor
        self.register_buffer('mapcbsg', mapcbsg)
        _init_conv_params(self, in_channels, out_channels, kernel_size, bias, weight_ext, bias_ext)

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


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

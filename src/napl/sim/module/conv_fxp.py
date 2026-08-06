from napl.utils import rshift_offset
from napl.sim.base import napl_base
from napl.sim.module._shared import _init_conv_params, _conv2d_binary, _linear_fxp_fn

# Single source for every optional key: the signature default and the per-key fallback.
_DEFAULT_CONFIG = {
    'widthi': 8,
    'quantilei': 1,
    'widthw': 8,
    'quantilew': 1,
    'rounding': 'round',
}


class conv_fxp(napl_base):
    r"""Apply a trainable fixed-point approximation of ``torch.nn.Conv2d``.

    Use this single-shot layer for quantization-aware convolution with
    ``groups=1`` and zero padding. The target is

    .. math::

       y = \mathrm{conv2d}(x, W) + b,

    evaluated with inputs and weights quantized to **widthi** and **widthw**
    signed fixed-point bits,

    .. math::

       \tilde{y} = \mathrm{conv2d}(Q_i(x), Q_w(W)) + b,

    where :math:`Q` rounds an operand onto its fixed-point grid and clamps it to
    the signed range, so rounding and clamping are the only departures from the
    target.

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
                 config=_DEFAULT_CONFIG):
        """Configure the convolution geometry and fixed-point approximation.

        .. container:: api-parameter-list

            **Parameters:**

            - **in_channels** – Number of input channels.
            - **out_channels** – Number of output channels.
            - **kernel_size** – Kernel size accepted as an integer or pair.
            - **stride** – Convolution stride; the default is ``1``.
            - **padding** – Symmetric zero padding; the default is ``0``.
            - **dilation** – Kernel dilation; the default is ``1``.
            - **bias** – Create a trainable bias when ``True``; the default is ``True``.
            - **weight_ext** – Optional initial weight shaped ``(out_channels, in_channels, kernel_height, kernel_width)``; the default is ``None``.
            - **bias_ext** – Optional initial bias shaped ``(out_channels,)``; the default is ``None``.
            - **config** – Configuration mapping. Omitted keys fall back to the same defaults.

              - **widthi**: Fixed-point width for input values; the default is ``8``.
              - **quantilei**: Input-magnitude scaling quantile; the default is ``1``.
              - **widthw**: Fixed-point width for weight values; the default is ``8``.
              - **quantilew**: Weight-magnitude scaling quantile; the default is ``1``.
              - **rounding**: Dynamic-scaling rounding mode; the default is ``"round"``.
              - **name**: Optional instance label.
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
        #: Fixed-point width used for input values.
        self.widthi = cfg['widthi']
        #: Fixed-point width used for weight values.
        self.widthw = cfg['widthw']
        #: Input-magnitude quantile used to choose the scaling shift.
        self.quantilei = cfg['quantilei']
        #: Weight-magnitude quantile used to choose the scaling shift.
        self.quantilew = cfg['quantilew']
        #: Rounding mode applied to the log2 magnitude that sets the scaling shift.
        self.rounding = cfg['rounding'].lower()
        #: Largest positive input magnitude represented by the quantized kernel.
        self.max_abs_i = 2 ** (self.widthi - 1)
        #: Largest positive weight magnitude represented by the quantized kernel.
        self.max_abs_w = 2 ** (self.widthw - 1)
        _init_conv_params(self, in_channels, out_channels, kernel_size, bias, weight_ext, bias_ext)

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


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

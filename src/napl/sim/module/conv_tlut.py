import math

from napl.sim.base import napl_base
from napl.sim.module._shared import (
    _init_conv_params,
    _conv2d_binary,
    _linear_tlut_fxpfxp_fn,
    _linear_tlut_fxpfp_fn,
    _linear_tlut_fpfp_fn,
    _TLUT_FP_WIDTH,
)
from loguru import logger

# Single source for every optional key: the signature default and the per-key fallback.
_DEFAULT_CONFIG = {
    'temporal': 'i',
    'widtht': 4,
    'formati': 'fxp',
    'widthi': 8,
    'quantilei': 1,
    'formatw': 'fxp',
    'widthw': 8,
    'quantilew': 1,
    'cycle': None,
    'rounding': 'round',
}


class conv_tlut(napl_base):
    r"""Apply a temporal-LUT approximation of ``torch.nn.Conv2d``.

    Use this single-shot trainable layer to decompose either convolution inputs or
    weights into temporal digits. The ``(formati, formatw)`` pair selects the
    ``fxpfxp``, ``fxpfp``, or ``fpfp`` arithmetic path, and the layer supports
    ``groups=1`` and zero padding. The target is

    .. math::

       y = \mathrm{conv2d}(x, W) + b.

    The operand named by **temporal** is replaced by its temporal decomposition
    into **degree** digits of **widtht** bits. The decomposition recomposes that
    operand exactly, so the departures from the target are the fixed-point
    truncation of the operands and the per-digit clamp that applies when
    **cycle** is below its cap. Gradients use a straight-through estimator.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import conv_tlut

        layer = conv_tlut(1, 2, 3, padding=1,
                          config={"temporal": "i", "widtht": 4,
                                  "formati": "fxp", "widthi": 8,
                                  "formatw": "fxp", "widthw": 8})
        output = layer(torch.zeros(1, 1, 4, 4))

    .. container:: api-references

        .. rubric:: References

        *Carat: Unlocking Value-Level Parallelism for Multiplier-Free GEMMs*, ASPLOS, 2024.

        *T-MAC: Temporal Multiplication with Accumulation*, Young Architect Workshop, 2022.
    """
    streaming = False


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bias=True, weight_ext=None, bias_ext=None,
                 config=_DEFAULT_CONFIG):
        """Configure convolution geometry and temporal decomposition.

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

              - **temporal**: Operand decomposed into temporal digits, ``"i"``/``"input"`` or ``"w"``/``"weight"``; the default is ``"i"``.
              - **widtht**: Bits per temporal digit; the default is ``4``.
              - **formati**: Input format, ``"fxp"`` or one of ``"bfloat16"``, ``"float16"``, ``"float32"``; the default is ``"fxp"``.
              - **widthi**: Fixed-point width for input values; the default is ``8``.
              - **quantilei**: Input-magnitude scaling quantile; the default is ``1``.
              - **formatw**: Weight format, with the same choices as **formati**; the default is ``"fxp"``.
              - **widthw**: Fixed-point width for weight values; the default is ``8``.
              - **quantilew**: Weight-magnitude scaling quantile; the default is ``1``.
              - **cycle**: Active cycles, capped at ``2 ** widtht``, where ``None`` selects the cap; the default is ``None``.
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
        #: Operand decomposed into temporal digits.
        self.temporal = cfg['temporal'].lower()
        #: Number of bits represented by each temporal digit.
        self.widtht = cfg['widtht']
        #: Numeric format used for input values.
        self.formati = cfg['formati'].lower()
        #: Numeric format used for weight values.
        self.formatw = cfg['formatw'].lower()
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
        if self.temporal not in ('i', 'input', 'w', 'weight'):
            message = f"conv_tlut 'temporal' must be one of ['i','input','w','weight'], got {self.temporal}."
            logger.error(message)
            raise AssertionError(message)

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
        cycle_cfg = cfg['cycle']
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

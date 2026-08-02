import torch
import math

from napl.utils import *
from napl.sim.base import napl_base
from napl.sim.module.linear import (_linear_fxp_fn, _linear_hub_fn,
                                _linear_tlut_fxpfxp_fn, _linear_tlut_fxpfp_fn, _linear_tlut_fpfp_fn,
                                _build_hub_map, _TLUT_FP_WIDTH)
from loguru import logger


def _init_conv_params(module, in_channels, out_channels, kernel_size, bias, weight_ext, bias_ext):
    """Give ``module`` ``nn.Conv2d``-style trainable parameters."""
    kh, kw = num2tuple(kernel_size)
    module.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
    torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    if bias:
        module.bias = torch.nn.Parameter(torch.empty(out_channels))
        bound = 1 / math.sqrt(in_channels * kh * kw)
        torch.nn.init.uniform_(module.bias, -bound, bound)
    else:
        module.bias = None
    if weight_ext is not None:
        assert tuple(weight_ext.shape) == (out_channels, in_channels, kh, kw), \
            logger.error(f'weight_ext shape {tuple(weight_ext.shape)} != {(out_channels, in_channels, kh, kw)}.')
        module.weight.data = weight_ext.clone().type(module.weight.dtype)
    if bias and bias_ext is not None:
        assert tuple(bias_ext.shape) == (out_channels,), \
            logger.error(f'bias_ext shape {tuple(bias_ext.shape)} != ({out_channels},).')
        module.bias.data = bias_ext.clone().type(module.bias.dtype)


def _conv2d_binary(input, weight, bias, kernel_size, stride, padding, dilation, linear_fn):
    """
    Run a binary-domain conv2d by im2col + a 2D linear kernel + fold: unfold the input to
    patches, apply ``linear_fn(patches_2d, weight_2d)`` through a binary linear autograd
    Functions), then fold the result back to NCHW. Bias is added after folding. The unfold/
    fold are differentiable, so the linear Function's STE gradient flows through to weight
    and input.
    """
    out_hw = conv2d_output_shape((input.size(2), input.size(3)), kernel_size=kernel_size,
                                 dilation=dilation, pad=padding, stride=stride)
    im2col = torch.nn.functional.unfold(input, kernel_size, dilation, padding, stride)
    inp2d = im2col.transpose(1, 2).reshape(-1, im2col.size(1))
    w2d = weight.view(weight.size(0), -1)
    mm = linear_fn(inp2d, w2d)
    mm = mm.reshape(input.size(0), -1, mm.size(-1)).transpose(1, 2)
    out = mm.reshape(input.size(0), mm.size(1), out_hw[0], out_hw[1])
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


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


class conv(napl_base):
    """Apply a rate-coded unary convolution one timestep at a time.

    Use this layer when the input is an NCHW spike stream and weights should be
    encoded on a separate number-sequence dimension. Each timestep the weights are encoded
    into spikes on a distinct RNG dimension, multiplied (XNOR bipolar / AND unipolar) with the
    im2col'd input patches, and the partial products summed by a scaled unary adder, then
    folded back to NCHW. The decoded output represents
    ``conv2d(x, W) + b`` divided by ``scale``, which defaults to
    ``in_channels * kh * kw + has_bias``, to stay in the unary range. Bipolar zero-padding uses a
    decorrelated rate-0.5 pad stream (a separate pad encoder), not a deterministic toggle.
    It supports rate-coded weights, ``groups=1``, and zero padding only.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import conv

        layer = conv(torch.zeros(2, 1, 3, 3), padding=1,
                     config={"polarity": "bipolar", "timestep": 4,
                             "generator": "sobol"})
        output_spike = layer(torch.ones(1, 1, 4, 4))
    """


    def __init__(self, weight, bias=None, stride=1, padding=0, dilation=1,
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol',
                         'dim': 2, 'scale': None, 'width': 12}):
        """Construct the streaming convolution from external numeric parameters.

        Args:
            weight: Numeric tensor shaped
                ``(out_channels, in_channels, kernel_height, kernel_width)``.
            bias: Optional numeric tensor shaped ``(out_channels,)``. Defaults to
                ``None``.
            stride: Convolution stride. Defaults to ``1``.
            padding: Symmetric zero padding. Defaults to ``0``.
            dilation: Kernel dilation. Defaults to ``1``.
            config: Configuration mapping with **polarity** (default
                ``"bipolar"``), **timestep** (default ``256``), **generator**
                (default ``"sobol"``), **dim** (weight Sobol dimension, default
                ``2``), **scale** (default ``None``, meaning fan-in plus bias),
                and **width** (accumulator width, default ``12``). **name** is an
                optional instance label and defaults to ``None``.

        **width** must satisfy ``2 ** (width - 1) >= fan_in + has_bias``. Bias and
        bipolar padding use the next number-sequence dimensions.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        from napl.sim.operation import add_any
        from napl.sim.module.encoder import encoder

        assert weight.dim() == 4, logger.error(f'conv weight must be 4D (out,in,kh,kw), got {tuple(weight.shape)}.')
        #: Trainable numeric convolution kernel encoded into a spike stream.
        self.weight = torch.nn.Parameter(weight)
        #: Optional trainable numeric bias encoded on its own sequence.
        self.bias = torch.nn.Parameter(bias) if bias is not None else None
        #: Number of convolution output channels.
        self.out_channels = weight.shape[0]
        #: Number of convolution input channels.
        self.in_channels = weight.shape[1]
        #: Spatial height and width of the convolution kernel.
        self.kernel_size = (weight.shape[2], weight.shape[3])
        #: Spatial step between adjacent convolution windows.
        self.stride = stride
        #: Spacing between kernel elements.
        self.dilation = dilation
        #: Symmetric padding represented as a height-width pair.
        self.padding = num2tuple(padding)
        #: Whether an encoded bias contributes to each output sum.
        self.has_bias = bias is not None
        #: Number of weight products in one convolution output.
        self.K = weight[0].numel()
        #: Unary-adder fan-in, including the bias when present.
        self.entry = self.K + (1 if self.has_bias else 0)
        scale = config.get('scale', None)
        #: Divisor implemented by the streaming unary adder.
        self.scale = self.entry if scale is None else scale

        width = config.get('width', 12)
        assert 2 ** (width - 1) >= self.entry, logger.error(
            f'conv accumulator width <{width}> too small for fan-in <{self.entry}>: '
            f'2**(width-1) must be >= entry or partial sums saturate. Increase width.')
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'conv decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim. Use a sobol-family generator.')

        dim = config.get('dim', 2)
        cfg = {'polarity': self.polarity, 'timestep': config['timestep'], 'generator': config['generator']}
        #: Encoder that converts the numeric convolution kernel to spikes.
        self.w_encoder = encoder({**cfg, 'dim': dim})
        #: Streaming unary adder that reduces each convolution product count.
        self.acc = add_any({'polarity': self.polarity, 'scale': self.scale, 'width': width})
        if self.has_bias:
            #: Encoder that converts the optional numeric bias to spikes.
            self.b_encoder = encoder({**cfg, 'dim': dim + 1})
        # Bipolar zero-padding uses a decorrelated rate-0.5 stream.
        if self.polarity == 'bipolar' and self.padding != (0, 0):
            #: Encoder supplying a decorrelated bipolar-zero padding stream.
            self.pad_encoder = encoder({**cfg, 'dim': dim + 2})
            # Python float pad bits avoid a device-to-host synchronization in F.pad.
            #: Period of the precomputed padding spike sequence.
            self.pad_len = self.pad_encoder.len
            pad_seq = torch.gt(torch.tensor(0.5, dtype=self.ntype),
                               self.pad_encoder.num_seq.detach()).type(self.stype)
            #: Precomputed scalar padding spikes indexed by timestep.
            self.pad_bits = [float(b) for b in pad_seq.tolist()]


    def _reset(self):
        """Reset state owned directly by the convolution.

        This class has no extra local state. The inherited ``reset()`` method
        resets the registered encoders and unary adder.
        """
        pass


    def forward(self, input_spike):
        """Process one NCHW input-spike timestep.

        Args:
            input_spike: ``0``/``1`` tensor shaped
                ``(batch, in_channels, height, width)``.

        Returns:
            Output spike tensor shaped
            ``(batch, out_channels, output_height, output_width)``.

        The call advances the convolution and its streaming children and updates
        the unary-adder accumulator. External weight and bias tensors are not
        modified.
        """
        ph, pw = self.padding
        out_hw = conv2d_output_shape((input_spike.size(2), input_spike.size(3)), kernel_size=self.kernel_size,
                                     dilation=self.dilation, pad=self.padding, stride=self.stride)
        # unfold requires floating input; converting 0/1 spikes is exact.
        xf = input_spike.type(self.ntype)
        if self.polarity == 'bipolar' and self.padding != (0, 0):
            # A decorrelated rate-0.5 pad stream represents bipolar zero.
            pad_bit = self.pad_bits[(self.timestep_cur - 1) % self.pad_len]
            xf = torch.nn.functional.pad(xf, (pw, pw, ph, ph), value=pad_bit)
            im2col = torch.nn.functional.unfold(xf, self.kernel_size, self.dilation, 0, self.stride)
        else:
            im2col = torch.nn.functional.unfold(xf, self.kernel_size, self.dilation, self.padding, self.stride)
        w_spike = self.w_encoder(self.weight.view(self.out_channels, -1))
        wf = w_spike.type(self.ntype)
        # Small integer AND/XNOR counts are exact in the floating accumulator.
        psum = torch.matmul(wf, im2col)
        if self.polarity == 'bipolar':
            psum = 2 * psum - im2col.sum(1, keepdim=True) - wf.sum(-1).unsqueeze(-1) + self.K
        if self.has_bias:
            psum = psum + self.b_encoder(self.bias).type(self.ntype).unsqueeze(-1)
        acc = self.acc(psum, entry=self.entry, dim=None)
        return acc.reshape(input_spike.size(0), acc.size(1), out_hw[0], out_hw[1])

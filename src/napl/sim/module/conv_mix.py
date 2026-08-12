import torch

from napl.utils import num2tuple
from napl.sim.base import napl_base
from napl.sim.operation import encode
from napl.sim.module._shared import _check_acc_width, conv2d_output_shape
from .linear_mix import linear_mix
from loguru import logger


class conv_mix(napl_base):
    r"""Apply a rate-coded unary convolution one timestep at a time.

    Use this layer when the input is an NCHW spike stream and weights should be
    encoded on a separate number-sequence dimension. The decoded output rate
    represents the convolution divided by **scale**, which defaults to the kernel
    fan-in plus the bias, keeping the result inside the unary range,

    .. math::

       y = \frac{\mathrm{conv2d}(x, W) + b}{s}.

    Bipolar zero padding draws a rate-``0.5`` spike from a separate pad encoder,
    so the padding decodes to ``0`` without correlating with the weight stream.
    The layer supports rate-coded weights, ``groups=1``, and zero padding only.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.module import conv_mix

        layer = conv_mix(torch.zeros(2, 1, 3, 3), padding=1,
                     config={"polarity": "bipolar", "timestep": 4,
                             "generator": "sobol"})
        output_spike = layer(torch.ones(1, 1, 4, 4))
    """
    #: The weight, bias, and pad encoders are held inside the layer, so the RTL
    #: counterpart encodes those operands itself from held numeric codes
    #: instead of taking them as spikes from a shared encoder.
    internal_encode = True


    def __init__(self, weight, bias=None, stride=1, padding=0, dilation=1,
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol',
                         'dim': 2, 'scale': None, 'width': 12}):
        """Construct the streaming convolution from external numeric parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **weight** – Numeric tensor shaped ``(out_channels, in_channels, kernel_height, kernel_width)``.
            - **bias** – Optional numeric tensor shaped ``(out_channels,)``; the default is ``None``.
            - **stride** – Convolution stride; the default is ``1``.
            - **padding** – Symmetric zero padding; the default is ``0``.
            - **dilation** – Kernel dilation; the default is ``1``.
            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **timestep**: Weight-encoder stream length; the default is ``256``.
              - **generator**: Number-sequence generator name; the default is ``"sobol"``.
              - **dim**: One-based weight Sobol dimension, with the bias on ``dim + 1`` and the bipolar pad stream on ``dim + 2``; the default is ``2``.
              - **scale**: Output divisor, where ``None`` uses the fan-in plus bias; the default is ``None``.
              - **width**: Signed accumulator width, which must satisfy ``2 ** (width - 1) - 1 >= (scale - grid) + delta_max``, where ``delta_max`` is the largest per-timestep accumulator step (``entry`` when unipolar, ``(entry + scale) / 2`` when bipolar, with ``entry = fan_in + has_bias``) and ``grid`` is the accumulator step (``0.5`` when bipolar with odd ``entry - scale``, else ``1``); the default is ``12``. This bound is static for ``scale >= entry``; for ``scale < entry`` the width must also satisfy ``2 ** (width - 1) > entry``, a minimum burst-headroom floor rather than a safety bound, since the accumulator then drains by at most ``scale`` per timestep and correctness is conditional on the long-run mean inflow staying below ``scale`` (see :class:`add_scale`).
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['dim', 'scale', 'width'], polarity_required=True)

        if weight.dim() != 4:
            message = f'conv_mix weight must be 4D (out,in,kh,kw), got {tuple(weight.shape)}.'
            logger.error(message)
            raise AssertionError(message)
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

        width = _check_acc_width('conv_mix', config.get('width', 12), self.entry,
                                 self.scale, self.polarity)
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'conv_mix decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim. Use a sobol-family generator.')

        dim = config.get('dim', 2)
        cfg = {'polarity': self.polarity, 'timestep': config['timestep'], 'generator': config['generator']}
        #: Inner product over one im2col patch, holding the weight and bias
        #: encoders and the streaming unary adder.
        self.core = linear_mix(
            weight.detach().reshape(self.out_channels, -1),
            bias,
            config={**cfg, 'dim': dim, 'scale': self.scale, 'width': width},
        )
        # This layer owns the parameters; the core reads them on every call.
        del self.core.weight, self.core.bias
        #: Trainable numeric convolution kernel encoded into a spike stream.
        self.weight = torch.nn.Parameter(weight)
        #: Optional trainable numeric bias encoded on its own sequence.
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

        # Bipolar zero-padding draws a decorrelated rate-0.5 stream from a separate pad encoder, not a deterministic toggle, so it does not correlate with the Sobol weight stream.
        if self.polarity == 'bipolar' and self.padding != (0, 0):
            #: Encoder supplying a decorrelated bipolar-zero padding stream.
            self.pad_encoder = encode({**cfg, 'dim': dim + 2})
            #: Period of the precomputed padding spike sequence.
            self.pad_len = self.pad_encoder.len
            pad_seq = torch.gt(torch.tensor(0.5, dtype=self.ntype),
                               self.pad_encoder.num_seq.detach()).type(self.stype)
            # Stands in for encode: converting the whole padding sequence to Python
            # floats once, because a per-timestep call would re-enter the encode
            # instance (self.pad_encoder) mid-stream.
            #: Precomputed scalar padding spikes indexed by timestep.
            self.pad_bits = [float(b) for b in pad_seq.tolist()]

        #: Flat gather indices that build the im2col patch layout, rebuilt on demand.
        self._im2col_idx: torch.Tensor
        self.register_buffer('_im2col_idx', None, persistent=False)
        #: Input shape and device the cached gather indices were built for.
        self._im2col_key = None
        #: Spatial output shape cached with the gather indices.
        self._out_hw = None

        # Encoders and the unary adder are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming layer.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Drop the cached convolution geometry.

        The gather indices and output shape are rebuilt on the next call. The
        inherited ``reset()`` method resets the linear core and pad encoder.
        """
        self._im2col_idx = None
        self._im2col_key = None
        self._out_hw = None


    def forward(self, input):
        """Process one NCHW input-spike timestep.

        Args:
            input: ``0``/``1`` tensor shaped
                ``(batch, in_channels, height, width)``.

        Returns:
            Output spike tensor shaped
            ``(batch, out_channels, output_height, output_width)``.

        The call advances the convolution and linear core and updates the
        unary-adder accumulator. External weight and bias tensors are not
        modified. Cached gather indices are rebuilt when the input geometry or
        device changes.
        """
        ph, pw = self.padding
        if self._im2col_key != (input.shape, input.device):
            self._build_im2col(input)
        # unfold requires floating input; converting 0/1 spikes is exact.
        xf = input.type(self.ntype)
        if self.polarity == 'bipolar' and self.padding != (0, 0):
            # A decorrelated rate-0.5 pad stream represents bipolar zero.
            pad_bit = self.pad_bits[(self.timestep_cur - 1) % self.pad_len]
            xf = torch.nn.functional.pad(xf, (pw, pw, ph, ph), value=pad_bit)
            inp = xf.reshape(xf.size(0), -1).index_select(1, self._im2col_idx).view(-1, self.K)
        else:
            im2col = torch.nn.functional.unfold(xf, self.kernel_size, self.dilation, self.padding, self.stride)
            inp = im2col.transpose(1, 2).reshape(-1, self.K)

        self.core.weight = self.weight.reshape(self.out_channels, -1)
        self.core.bias = None if self.bias is None else self.bias.data
        acc = self.core(inp)
        return acc.view(input.size(0), -1, acc.size(-1)).transpose(1, 2) \
                  .reshape(input.size(0), acc.size(-1), *self._out_hw)


    @property
    def w_encoder(self):
        """Weight encoder owned by the composed linear core."""
        return self.core.w_encoder


    @property
    def b_encoder(self):
        """Optional bias encoder owned by the composed linear core."""
        return self.core.b_encoder


    @property
    def acc(self):
        """Streaming unary adder owned by the composed linear core."""
        return self.core.acc


    def _build_im2col(self, input):
        ph, pw = self.padding
        self._out_hw = conv2d_output_shape(
            (input.size(2), input.size(3)),
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            pad=self.padding,
            stride=self.stride,
        )
        c = input.size(1)
        hp = input.size(2) + 2 * ph
        wp = input.size(3) + 2 * pw
        # Build exact float64 indices on CPU because MPS lacks float64.
        ar = torch.arange(c * hp * wp, dtype=torch.float64).view(1, c, hp, wp)
        unfolded = torch.nn.functional.unfold(
            ar, self.kernel_size, self.dilation, 0, self.stride
        )
        # Output-position-major indices map a flat gather to (positions, K).
        self._im2col_idx = unfolded.view(self.K, -1).t().contiguous().long().view(-1) \
            .to(input.device)
        self._im2col_key = (input.shape, input.device)

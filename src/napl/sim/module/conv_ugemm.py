import torch

from napl.utils import conv2d_output_shape, num2tuple
from napl.sim.base import napl_base, hw_params
from napl.sim.operation import add_any, mul_ugemm
from loguru import logger


class conv_ugemm(napl_base):
    r"""Apply streaming unary convolution with conditional spike generation.

    Use this layer when input-driven uGEMM weight streams are preferred over the
    free-running weight encoder used by :class:`conv`. Weight spikes come from
    :class:`mul_ugemm` conditional spike generation, where each weight-stream
    index advances only when its input spike is ``1``, and bipolar mode adds an
    input-``0`` path on a separate index. The decoded output rate
    represents the convolution divided by **scale**, which defaults to the kernel
    fan-in plus the bias,

    .. math::

       y = \frac{\mathrm{conv2d}(x, W) + b}{s}.

    Bipolar zero padding alternates ``0`` and ``1`` each timestep, a
    deterministic rate-``0.5`` stream, where :class:`conv` instead uses a
    decorrelated pad encoder. The layer is rate-coded, supports ``groups=1`` and
    zero padding, and implements only the scaled UnarySim ``FSUConv2duGEMM``
    mode.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import conv_ugemm

        layer = conv_ugemm(torch.zeros(2, 1, 3, 3), padding=1,
                           config={"polarity": "bipolar", "timestep": 4,
                                   "generator": "sobol"})
        output_spike = layer(torch.ones(1, 1, 4, 4))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """
    #: Encoding advances conditionally on data, so the RTL counterpart holds
    #: its own encoder instead of sharing an external one.
    internal_encode = True


    def __init__(self, weight, bias=None, stride=1, padding=0, dilation=1,
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol',
                         'scale': None, 'width': 12}):
        """Construct the streaming CSG convolution.

        .. container:: api-parameter-list

            **Parameters:**

            - **weight** – Numeric convolution weight shaped ``(out_channels, in_channels, kernel_height, kernel_width)``.
            - **bias** – Optional numeric tensor shaped ``(out_channels,)``; the default is ``None``.
            - **stride** – Convolution stride; the default is ``1``.
            - **padding** – Symmetric zero padding; the default is ``0``.
            - **dilation** – Kernel dilation; the default is ``1``.
            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **timestep**: Positive stream length; the default is ``256``.
              - **generator**: Number-sequence generator name; the default is ``"sobol"``.
              - **scale**: Output divisor, where ``None`` uses the fan-in plus bias; the default is ``None``.
              - **width**: Signed accumulator width, which must satisfy ``2 ** (width - 1) > fan_in + has_bias``; the default is ``12``.
              - **name**: Optional instance label.

        Weight and bias are updatable only by an in-place write, such as
        ``with torch.no_grad(): layer.weight.fill_(1.0)``. Each timestep reads
        their current values, so the write takes effect on the next call,
        without a ``reset()``.

        .. warning::

            Optimizers silently skip these ``Parameter`` objects. The spike
            comparison is not differentiable, so no gradient ever reaches them,
            ``.grad`` stays ``None``, and ``SGD.step()`` leaves the values
            bit-identical.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['scale', 'width'], polarity_required=True)

        if weight.dim() != 4:
            message = f'conv_ugemm weight must be 4D (out,in,kh,kw), got {tuple(weight.shape)}.'
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
        self.K = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        #: Parallel-count fan-in, including the bias when present.
        self.entry = self.K + (1 if self.has_bias else 0)
        scale = config.get('scale', None)
        #: Divisor implemented by the streaming unary adder.
        self.scale = self.entry if scale is None else scale
        self._is_bipolar = (self.polarity == 'bipolar')

        width = config.get('width', 12)
        if 2 ** (width - 1) <= self.entry:
            message = (
                f'conv_ugemm accumulator width <{width}> too small for fan-in <{self.entry}>: '
                f'2**(width-1) must be > entry or partial sums saturate. Increase width.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Requested number of output-spike timesteps in the stream.
        self.timestep = config['timestep']
        if self.timestep <= 0:
            message = f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.'
            logger.error(message)
            raise AssertionError(message)
        # Weight and bias CSG paths share one RNG sequence.
        #: Conditional spike generator holding the weight sequence and its per-input indices.
        self.mul = mul_ugemm({'polarity': self.polarity, 'timestep': self.timestep,
                              'generator': config['generator']})
        #: Power-of-two period of the conditional-generator sequence.
        self.len = self.mul.len

        #: Externally updatable numeric weight tensor converted to spike probabilities on use.
        self.weight = torch.nn.Parameter(weight)
        #: Optional externally updatable numeric bias converted to spike probabilities on use.
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

        #: Streaming unary adder that reduces each convolution product count.
        self.acc = add_any({'polarity': self.polarity, 'scale': self.scale, 'width': width})

        #: Flat gather indices that build the im2col patch layout, rebuilt on demand.
        self._im2col_idx: torch.Tensor
        self.register_buffer('_im2col_idx', None, persistent=False)
        #: Input shape and device the cached gather indices were built for.
        self._im2col_key = None
        #: Spatial output shape cached with the gather indices.
        self._out_hw = None

        # Conditional generation and the adder are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming layer.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input_spike': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_spike': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Drop the cached convolution geometry.

        The gather indices and output shape are rebuilt on the next call. The
        inherited ``reset()`` method restarts the conditional spike generator,
        which owns the per-input sequence indices, and the unary adder.
        """
        self._im2col_idx = None
        self._im2col_key = None
        self._out_hw = None


    def forward(self, input_spike):
        """Process one NCHW input-spike timestep.

        Args:
            input_spike: ``0``/``1`` tensor shaped
                ``(batch, in_channels, height, width)``.

        Returns:
            Output spike tensor shaped
            ``(batch, out_channels, output_height, output_width)``.

        The call advances the per-patch conditional RNG indices, the unary
        adder, and ``timestep_cur``. Weight and bias spike probabilities are
        recomputed from the current parameters on every call. Cached gather
        indices are rebuilt when the input geometry or device changes.
        """
        ph, pw = self.padding
        if self._im2col_key != (input_spike.shape, input_spike.device):
            self._build_im2col(input_spike)
        # RNG comparison arithmetic requires floating input.
        xf = input_spike.type(self.ntype)
        if self.padding != (0, 0):
            # Bipolar zero-padding alternates 0 and 1, starting with 0.
            pad_bit = float((self.timestep_cur - 1) % 2) if self._is_bipolar else 0.0
            xf = torch.nn.functional.pad(xf, (pw, pw, ph, ph), value=pad_bit)
        inp = xf.reshape(xf.size(0), -1).index_select(1, self._im2col_idx).view(-1, self.K)

        # The patch spike broadcasts over output channels, so each patch position
        # keeps one sequence index shared by the whole weight row.
        product = self.mul(inp.unsqueeze(1), self.weight.reshape(self.out_channels, -1))
        psum = product.type(self.ntype).sum(-1)

        if self.has_bias:
            # The bias stream advances once per timestep on the shared RNG.
            b_prob = ((self.bias + 1) / 2 if self._is_bipolar else self.bias).type(self.ntype)
            psum += torch.gt(b_prob,
                             self.mul.num_seq[(self.timestep_cur - 1) % self.len]).type(self.ntype)

        acc = self.acc(psum, entry=self.entry, dim=None)
        return acc.view(input_spike.size(0), -1, acc.size(-1)).transpose(1, 2) \
                  .reshape(input_spike.size(0), acc.size(-1), *self._out_hw)


    def _build_im2col(self, input_spike):
        ph, pw = self.padding
        self._out_hw = conv2d_output_shape((input_spike.size(2), input_spike.size(3)),
                                           kernel_size=self.kernel_size, dilation=self.dilation,
                                           pad=self.padding, stride=self.stride)
        c, hp, wp = input_spike.size(1), input_spike.size(2) + 2 * ph, input_spike.size(3) + 2 * pw
        # Build exact float64 indices on CPU because MPS lacks float64.
        ar = torch.arange(c * hp * wp, dtype=torch.float64).view(1, c, hp, wp)
        u = torch.nn.functional.unfold(ar, self.kernel_size, self.dilation, 0, self.stride)
        # L-major indices map a flat gather to the (P, K) patch layout.
        self._im2col_idx = u.view(self.K, -1).t().contiguous().long().view(-1).to(input_spike.device)
        self._im2col_key = (input_spike.shape, input_spike.device)

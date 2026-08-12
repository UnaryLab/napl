import torch

from napl.utils import num2tuple
from napl.sim.base import napl_base
from napl.sim.operation import encode
from napl.sim.module._shared import conv2d_output_shape
from .linear_gaines import linear_gaines
from loguru import logger


class conv_gaines(napl_base):
    r"""Apply a streaming Gaines ``gMUL + gADD`` convolution one timestep at a time.

    Use this layer when the input is an NCHW spike stream and the inner product
    over each kernel window should use Gaines multiplication and addition rather
    than the scaled accumulator of :class:`conv_mix`. Every position of the
    flattened kernel carries its own rate-coded weight sequence, taken from a
    distinct Sobol dimension, and a Gaines adder reduces the products, with
    ``entry = in_channels * kernel_height * kernel_width + has_bias``:

    - ``scaled=True``: MUX-based scaled addition over ``entry`` inputs
      (``entry`` must be a power of two). The decoded output represents

      .. math::

         y = \frac{\mathrm{conv2d}(x, W) + b}{\mathrm{entry}}.

    - ``scaled=False``: unipolar mode OR-reduces the product bits, approaching
      ``min(1, sum)``. Non-scaled bipolar addition is not supported.

    Bipolar zero padding draws a rate-``0.5`` spike from a separate pad encoder,
    so the padding decodes to ``0`` without correlating with the weight streams.
    The layer supports rate-coded weights, ``groups=1``, and zero padding only.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.module import conv_gaines

        layer = conv_gaines(torch.zeros(2, 2, 2, 2), padding=1,
                            config={"polarity": "bipolar", "timestep": 256,
                                    "generator": "sobol"})
        output_spike = layer(torch.ones(1, 2, 4, 4))

    .. container:: api-references

        .. rubric:: References

        *Stochastic Computing Systems*, Advances in Information Systems Science, 1969.
    """
    #: One threshold sequence per kernel position is held inside the layer, so the
    #: RTL counterpart generates the weight and bias streams itself from held
    #: numeric codes instead of taking them as spikes from a shared encoder.
    internal_encode = True


    def __init__(self, weight, bias=None, stride=1, padding=0, dilation=1,
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol',
                         'dim': 2, 'scaled': True}):
        """Construct the streaming Gaines convolution from external numeric parameters.

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
              - **dim**: First weight sequence dimension, with one dimension per kernel position above it, the bias on ``dim + K``, the scaled adder's MUX select sequence on ``dim + K + 1`` and the bipolar pad stream on ``dim + K + 2``, where ``K`` is the kernel fan-in; the default is ``2``.
              - **scaled**: Use MUX-select scaled addition, which picks one input per timestep by a Sobol-derived index, when ``True``; the default is ``True``.
              - **name**: Optional instance label.

        In scaled mode, ``in_channels * kernel_height * kernel_width + has_bias``
        must be a power of two and at least ``2``. Non-scaled mode accepts
        unipolar data only.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['dim', 'scaled'], polarity_required=True)

        #: Whether the Gaines adder uses MUX-select scaled addition, selecting one
        #: input bit per timestep by a Sobol-derived index.
        self.scaled = config.get('scaled', True)
        if self.polarity == 'bipolar' and not self.scaled:
            message = 'Non-scaled Gaines addition in conv_gaines does not support bipolar data.'
            logger.error(message)
            raise AssertionError(message)

        if weight.dim() != 4:
            message = f'conv_gaines weight must be 4D (out,in,kh,kw), got {tuple(weight.shape)}.'
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
        #: Gaines-adder fan-in, including the bias when present.
        self.entry = self.K + (1 if self.has_bias else 0)

        # Report the fan-in constraint against the convolution geometry the caller
        # supplied, before the flattened core reports it against in_features.
        if self.scaled:
            if self.entry < 2:
                message = f'conv_gaines scaled mode needs entry >= 2, got {self.entry}.'
                logger.error(message)
                raise AssertionError(message)
            if self.entry & (self.entry - 1):
                message = f'conv_gaines scaled mode needs power-of-two entry, got {self.entry}.'
                logger.error(message)
                raise AssertionError(message)

        dim = config.get('dim', 2)
        cfg = {'polarity': self.polarity, 'timestep': config['timestep'],
               'generator': config['generator']}
        #: Inner product over one im2col patch, holding the weight and bias
        #: sequences and the Gaines adder.
        self.core = linear_gaines(
            weight.detach().reshape(self.out_channels, -1),
            bias,
            config={**cfg, 'dim': dim, 'scaled': self.scaled},
        )
        # This layer owns the parameters; the core reads them on every call.
        del self.core.weight, self.core.bias
        #: Trainable numeric convolution kernel encoded into a spike stream.
        self.weight = torch.nn.Parameter(weight)
        #: Optional trainable numeric bias encoded on its own sequence.
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

        # Gaines products are correlation sensitive, so bipolar zero padding uses a
        # decorrelated rate-0.5 stream rather than a deterministic 0/1 toggle.
        if self.polarity == 'bipolar' and self.padding != (0, 0):
            #: Encoder supplying a decorrelated bipolar-zero padding stream.
            self.pad_encoder = encode({**cfg, 'dim': dim + self.K + 2})
            #: Period of the precomputed padding spike sequence.
            self.pad_len = self.pad_encoder.len
            pad_seq = torch.gt(torch.tensor(0.5, dtype=self.ntype),
                               self.pad_encoder.num_seq.detach()).type(self.stype)
            # Python float pad bits avoid a device-to-host synchronization in F.pad.
            #: Precomputed scalar padding spikes indexed by timestep.
            self.pad_bits = [float(b) for b in pad_seq.tolist()]

        #: Flat gather indices that build the im2col patch layout, rebuilt on demand.
        self._im2col_idx: torch.Tensor
        self.register_buffer('_im2col_idx', None, persistent=False)
        #: Input shape and device the cached gather indices were built for.
        self._im2col_key = None
        #: Spatial output shape cached with the gather indices.
        self._out_hw = None

        # Encoders and the Gaines adder are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming layer.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Drop the cached convolution geometry.

        The gather indices and output shape are rebuilt on the next call. The
        inherited ``reset()`` method resets the Gaines core and pad encoder.
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

        The call advances the optional bias encoder, the Gaines adder, and
        ``timestep_cur``. Weight and bias spike probabilities are recomputed
        from the current parameters on every call. Cached gather indices are
        rebuilt when the input geometry or device changes.
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
            im2col = torch.nn.functional.unfold(xf, self.kernel_size, self.dilation,
                                                self.padding, self.stride)
            inp = im2col.transpose(1, 2).reshape(-1, self.K)

        self.core.weight = self.weight.reshape(self.out_channels, -1)
        self.core.bias = None if self.bias is None else self.bias.data
        acc = self.core(inp)
        return acc.view(input.size(0), -1, acc.size(-1)).transpose(1, 2) \
                  .reshape(input.size(0), acc.size(-1), *self._out_hw)


    @property
    def acc(self):
        """Accumulator owned by the composed linear core."""
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

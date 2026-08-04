import torch
import math

from napl.utils import conv2d_output_shape, num2tuple
from napl.sim.base import napl_base
from loguru import logger


class conv_ugemm(napl_base):
    """Apply streaming unary convolution with conditional spike generation.

    Use this layer when input-driven uGEMM weight streams are preferred over the
    free-running weight encoder used by :class:`conv`. It provides uGEMM-style
    conditional spike generation. Each weight-stream index advances when its
    input spike is ``1``, following the ``mul_csg`` rule, so products are
    input-driven. Per timestep, image-column input spikes gate the weight bits;
    bipolar mode adds the input-``0`` inverse path. A scaled unary adder emits
    the output spike and the decoded output represents
    ``(conv2d(x, W) + b) / scale``, with **scale** defaulting to
    ``in_channels * kernel_height * kernel_width + has_bias``. Bipolar
    zero-padding alternates ``0`` and ``1`` each timestep, a deterministic
    rate-``0.5`` stream, unlike :class:`conv`'s decorrelated pad
    encoder). It is rate-coded, supports ``groups=1`` and zero padding, and
    implements only the scaled UnarySim ``FSUConv2duGEMM`` mode.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import conv_ugemm

        layer = conv_ugemm(torch.zeros(2, 1, 3, 3), padding=1,
                           config={"polarity": "bipolar", "timestep": 4,
                                   "generator": "sobol"})
        output_spike = layer(torch.ones(1, 1, 4, 4))

    References
    ----------
    *uGEMM: Unary Computing Architecture for GEMM Applications*.
    """


    def __init__(self, weight, bias=None, stride=1, padding=0, dilation=1,
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol',
                         'scale': None, 'width': 12}):
        """Construct the streaming CSG convolution.

        Args:
            weight: Numeric convolution weight shaped
                ``(out_channels, in_channels, kernel_height, kernel_width)``.
            bias: Optional numeric tensor shaped ``(out_channels,)``. Defaults to
                ``None``.
            stride: Convolution stride. Defaults to ``1``.
            padding: Symmetric zero padding. Defaults to ``0``.
            dilation: Kernel dilation. Defaults to ``1``.
            config: Configuration mapping with **polarity** (default
                ``"bipolar"``), **timestep** (positive stream length, default
                ``256``), **generator** (default ``"sobol"``), **scale**
                (default ``None``, meaning fan-in plus bias), and **width**
                (accumulator width, default ``12``). **name** is an optional
                instance label and defaults to ``None``.

        **width** must satisfy ``2 ** (width - 1) > fan_in + has_bias``.
        Numeric weights and bias are converted to persistent spike probabilities.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # These imports stay local to avoid the module-operation import cycle.
        from napl.sim.operation import add_any
        from napl.sim.module.encoder import gen_num_seq

        assert weight.dim() == 4, logger.error(
            f'conv_ugemm weight must be 4D (out,in,kh,kw), got {tuple(weight.shape)}.')
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
        assert 2 ** (width - 1) > self.entry, logger.error(
            f'conv_ugemm accumulator width <{width}> too small for fan-in <{self.entry}>: '
            f'2**(width-1) must be > entry or partial sums saturate. Increase width.')

        #: Requested number of output-spike timesteps in the stream.
        self.timestep = config['timestep']
        assert self.timestep > 0, logger.error(
            f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        rng_width = math.ceil(math.log2(self.timestep))
        #: Power-of-two period of the conditional-generator sequence.
        self.len = 2 ** rng_width
        # Weight and bias CSG paths share one RNG sequence.
        #: Threshold sequence shared by weight and bias spike generation.
        self.num_seq: torch.Tensor
        self.register_buffer(
            'num_seq',
            gen_num_seq(config={'width': rng_width, 'generator': config['generator']}),
        )

        w_prob = (weight + 1) / 2 if self._is_bipolar else weight
        #: Flattened numeric weight probabilities used by conditional generation.
        self.w_prob: torch.Tensor
        self.register_buffer(
            'w_prob',
            w_prob.reshape(self.out_channels, -1).type(self.ntype).detach(),
        )
        if self.has_bias:
            b_prob = (bias + 1) / 2 if self._is_bipolar else bias
            #: Numeric bias probabilities used to generate the bias spike stream.
            self.b_prob: torch.Tensor
            self.register_buffer('b_prob', b_prob.type(self.ntype).detach())

        # Input-one and input-zero bits advance separate CSG indices.
        #: Per-input conditional-generator indices advanced by input-one spikes.
        self.w_idx: torch.Tensor
        self.register_buffer('w_idx', torch.zeros(1, dtype=torch.long))
        if self._is_bipolar:
            #: Per-input indices advanced by input-zero spikes in bipolar mode.
            self.w_idx_inv: torch.Tensor
            self.register_buffer('w_idx_inv', torch.zeros(1, dtype=torch.long))

        #: Streaming unary adder that reduces each convolution product count.
        self.acc = add_any({'polarity': self.polarity, 'scale': self.scale, 'width': width})
        self._im2col_key = None


    def _reset(self):
        """Reset the local conditional-generator indices.

        The input-one path index and, for bipolar streams, the input-zero path
        index return to scalar zero. Cached convolution geometry remains available.
        """
        self.w_idx.resize_(1).zero_()
        if self._is_bipolar:
            self.w_idx_inv.resize_(1).zero_()


    def forward(self, input_spike):
        """Process one NCHW input-spike timestep.

        Args:
            input_spike: ``0``/``1`` tensor shaped
                ``(batch, in_channels, height, width)``.

        Returns:
            Output spike tensor shaped
            ``(batch, out_channels, output_height, output_width)``.

        The call advances conditional RNG indices, the unary-adder state, and
        ``timestep_cur``. It may refresh cached gather indices when input geometry
        or device changes.
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
        inv = 1 - inp

        # Adding 2 disables CSG comparisons on input-zero lanes.
        rnd = self.num_seq[self.w_idx] + inv * 2
        psum = torch.gt(self.w_prob.unsqueeze(0), rnd.unsqueeze(1)).sum(-1, dtype=self.ntype)
        # Input-one indices expand once, then advance in place.
        if self.w_idx.shape == inp.shape:
            self.w_idx.add_(inp.type(torch.long))
        else:
            expanded = self.w_idx.add(inp.type(torch.long)).detach()
            self.w_idx.resize_as_(expanded).copy_(expanded)

        if self._is_bipolar:
            # Subtracting 2 disables inverse comparisons on input-one lanes.
            rnd_inv = self.num_seq[self.w_idx_inv] - inp * 2
            psum += torch.le(self.w_prob.unsqueeze(0), rnd_inv.unsqueeze(1)).sum(-1, dtype=self.ntype)
            if self.w_idx_inv.shape == inv.shape:
                self.w_idx_inv.add_(inv.type(torch.long))
            else:
                expanded = self.w_idx_inv.add(inv.type(torch.long)).detach()
                self.w_idx_inv.resize_as_(expanded).copy_(expanded)

        if self.has_bias:
            # The bias stream advances once per timestep on the shared RNG.
            psum += torch.gt(self.b_prob,
                             self.num_seq[(self.timestep_cur - 1) % self.len]).type(self.ntype)

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

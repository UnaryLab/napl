import torch
import math

from napl.utils import conv2d_output_shape, num2tuple
from napl.sim.base import napl_base
from loguru import logger


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

**width** must satisfy ``2 ** (width - 1) > fan_in + has_bias``. Bias and
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
        assert 2 ** (width - 1) > self.entry, logger.error(
            f'conv accumulator width <{width}> too small for fan-in <{self.entry}>: '
            f'2**(width-1) must be > entry or partial sums saturate. Increase width.')
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

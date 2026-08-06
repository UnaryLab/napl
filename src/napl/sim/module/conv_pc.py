import torch

from napl.utils import conv2d_output_shape, num2tuple
from napl.sim.base import napl_base, hw_params
from napl.sim.operation.encode import encode
from loguru import logger


class conv_pc(napl_base):
    r"""Return per-timestep parallel counts for a unary convolution.

    Use this streaming layer when downstream logic needs raw convolution product
    counts instead of a scaled output bitstream. It is the :class:`conv` partial
    sum without its scaled unary adder, and the convolution counterpart of
    :class:`linear_pc`. Unipolar counts are AND counts of the input and weight
    spikes, bipolar counts are XNOR counts, and each count lies in
    ``[0, entry]`` with
    ``entry = in_channels * kernel_height * kernel_width + has_bias``.

    The target is the count :math:`c_t` whose time average recovers the
    convolution,

    .. math::

       \frac{1}{T}\sum_{t=1}^{T} c_t = \mathrm{conv2d}(x, W) + b
       \ \ (\text{unipolar}),\qquad
       \frac{2}{T}\sum_{t=1}^{T} c_t - e = \mathrm{conv2d}(x, W) + b
       \ \ (\text{bipolar}),

    with :math:`e` the fan-in ``entry``. Weights and bias are encoded on RNG
    dimensions distinct from the input, so the operands are decorrelated, and
    bipolar zero padding uses a decorrelated rate-``0.5`` pad stream. This class
    matches UnarySim ``FSUConv2dPC`` and supports ``groups=1`` with zero padding
    only.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import conv_pc

        counter = conv_pc(torch.ones(2, 1, 3, 3), padding=1,
                          config={"polarity": "unipolar", "timestep": 4,
                                  "generator": "sobol"})
        count = counter(torch.ones(1, 1, 4, 4))

    The output is a parallel count rather than a spike stream, so it carries no
    encoding or polarity and appears in neither port map.

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """


    def __init__(self, weight, bias=None, stride=1, padding=0, dilation=1,
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol', 'dim': 2}):
        """Construct the counter from external numeric weights and bias.

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
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['dim'], polarity_required=True)

        if weight.dim() != 4:
            message = f'conv_pc weight must be 4D (out,in,kh,kw), got {tuple(weight.shape)}.'
            logger.error(message)
            raise AssertionError(message)
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
        #: Whether an encoded bias contributes to each output count.
        self.has_bias = bias is not None
        #: Number of weight products in one convolution output.
        self.K = weight[0].numel()
        #: Parallel-count fan-in, including the bias when present.
        self.entry = self.K + (1 if self.has_bias else 0)

        dim = config.get('dim', 2)
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'conv_pc decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim. Use a sobol-family generator.')

        cfg = {'polarity': self.polarity, 'timestep': config['timestep'], 'generator': config['generator']}
        #: Encoder that converts the numeric convolution kernel to spikes.
        self.w_encoder = encode({**cfg, 'dim': dim})
        if self.has_bias:
            #: Encoder that converts the optional numeric bias to spikes.
            self.b_encoder = encode({**cfg, 'dim': dim + 1})
        # Bipolar zero-padding uses a decorrelated rate-0.5 stream.
        if self.polarity == 'bipolar' and self.padding != (0, 0):
            #: Encoder supplying a decorrelated bipolar-zero padding stream.
            self.pad_encoder = encode({**cfg, 'dim': dim + 2})
            # Python float pad bits avoid a device-to-host synchronization in F.pad.
            #: Period of the precomputed padding spike sequence.
            self.pad_len = self.pad_encoder.len
            pad_seq = torch.gt(torch.tensor(0.5, dtype=self.ntype),
                               self.pad_encoder.num_seq.detach()).type(self.stype)
            #: Precomputed scalar padding spikes indexed by timestep.
            self.pad_bits = [float(b) for b in pad_seq.tolist()]

        # Encoding and the parallel count are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming counter.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input_spike': 'rc'}
        self.polarity_io = {'input_spike': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset state owned directly by the counter.

        This class has no extra local state. The inherited ``reset()`` method
        resets its registered encoders.
        """
        pass


    def forward(self, input_spike):
        """Count convolution spike products for one timestep.

        Args:
            input_spike: ``0``/``1`` NCHW tensor shaped
                ``(batch, in_channels, height, width)``.

        Returns:
            Numeric count tensor shaped
            ``(batch, out_channels, output_height, output_width)``. Values lie in
            ``[0, kernel_fan_in + has_bias]``.

        The call advances the counter and its encoders. It does not accumulate
        counts across timesteps.
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
        inp = im2col.transpose(1, 2).reshape(-1, im2col.size(1))
        w_spike = self.w_encoder(self.weight.view(self.out_channels, -1))
        wf = w_spike.type(self.ntype)
        pc = torch.matmul(inp, wf.t())
        if self.has_bias:
            # Bias contributes only to the input-one path.
            pc = pc + self.b_encoder(self.bias).type(self.ntype)
        if self.polarity == 'bipolar':
            # Bipolar XNOR count adds the input-zero AND path.
            pc = pc + torch.matmul(1 - inp, (1 - wf).t())
        out = pc.reshape(input_spike.size(0), -1, pc.size(-1)).transpose(1, 2)
        return torch.nn.functional.fold(out, out_hw, (1, 1))

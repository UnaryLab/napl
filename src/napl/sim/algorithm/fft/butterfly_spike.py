import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import add_any, decode, encode, mul_ugemm
from napl.sim.metric import accuracy
from loguru import logger


class butterfly_spike(napl_base):
    r"""
    Evaluate a radix-2 complex butterfly through streaming spike operations.

    Use this class for a progressively decoded unary FFT stage. Use
    :class:`butterfly_binary` when an exact binary-domain reference is needed.

    The target is the radix-2 decimation-in-time butterfly on complex inputs
    :math:`x_0`, :math:`x_1` and twiddle factor :math:`w`,

    .. math::

       y_0 = x_0 + w x_1,\qquad y_1 = x_0 - w x_1.

    Operands are encoded into bipolar spike streams and the outputs are read
    from a progressive decoder, so the returned values approximate the target
    divided by the adder scale from ``add_config``, within the
    stochastic-computing error of the encoded streams,

    .. math::

       \hat y_0 \approx \frac{x_0 + w x_1}{\mathit{scale}},\qquad
       \hat y_1 \approx \frac{x_0 - w x_1}{\mathit{scale}}.

    Only bipolar encoding is supported, because :math:`y_1 = x_0 - w x_1` is
    negative for positive operands and a unipolar stream cannot represent that.

    The ports carry numeric values rather than spike streams, since the class
    encodes its inputs and decodes its outputs internally.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.algorithm.fft import butterfly_spike

        codec = {'polarity': 'bipolar', 'timestep': 64, 'generator': 'sobol'}
        adder = {'polarity': 'bipolar', 'scale': 3, 'width': 4}
        operation = butterfly_spike(codec, codec, adder, {'polarity': 'bipolar'})
        x0 = (torch.tensor([0.5]), torch.tensor([0.25]))
        x1 = (torch.tensor([0.5]), torch.tensor([-0.25]))
        w = (torch.tensor([0.5]), torch.tensor([0.5]))
        for _ in range(64):
            y0r, y0i, y1r, y1i = operation(*x0, *x1, *w)
        print([round(y.item(), 4) for y in (y0r, y0i, y1r, y1i)])
        # [0.2812, 0.125, 0.0312, 0.0312]

    With :math:`w x_1 = (0.375, 0.125)`, the exact scaled outputs are
    :math:`y_0 / 3 = (0.2917, 0.125)` and :math:`y_1 / 3 = (0.0417, 0.0417)`.
    """


    def __init__(
            self,
            codec_config,
            mul_config,
            add_config,
            acc_config,
        ):
        """
        Configure the spike codecs, multiplier, adder, and error monitor.

        All four mappings are required and are passed to their corresponding
        NAPL components without merging defaults.

        .. container:: api-parameter-list

            **Parameters:**

            - **codec_config** – Shared encoder and decoder configuration.

              - **polarity**: Required stream encoding, which must be
                ``"bipolar"``.
              - **timestep**: Required positive maximum decoder run length.
              - **generator**: Required encoder number-sequence generator. The
                accepted values are ``"sobol"``, ``"lfsr"``, ``"sys"``,
                ``"rc"``, ``"tc"``, ``"rate"``, and ``"temporal"``.
              - **dim**: Sobol dimension; the default is ``1``.
              - **seed**: LFSR seed; the default is ``None``.
              - **taps**: LFSR feedback taps; the default is ``None``.
              - **name**: Optional component label; the default is ``None``.

            - **mul_config** – Conditional-spike multiplier configuration.

              - **polarity**: Required stream encoding, which must be
                ``"bipolar"``.
              - **timestep**: Required positive sequence length basis.
              - **generator**: Required number-sequence generator, with the same
                accepted values as ``codec_config``.
              - **name**: Optional component label; the default is ``None``.

            - **add_config** – Scaled-adder configuration.

              - **polarity**: Required stream encoding, which must be
                ``"bipolar"``.
              - **scale**: Required output carry scale.
              - **width**: Required signed accumulator width in bits.
              - **name**: Optional component label; the default is ``None``.

            - **acc_config** – Output accuracy-metric configuration.

              - **polarity**: Required stream encoding, which must be
                ``"bipolar"``.
              - **name**: Optional metric label; the default is ``None``.
        """
        super().__init__(codec_config, ['polarity', 'timestep', 'generator'],
                         optional_key_list=['width', 'dim', 'seed', 'taps'], polarity_required=True)
        # y1 = x0 - w x1 is negative for positive operands, which a unipolar stream
        # cannot represent, so the subtraction path requires bipolar encoding.
        if self.polarity != 'bipolar':
            message = f'Invalid polarity: <{self.polarity}>; legal values: <[\'bipolar\']>.'
            logger.error(message)
            raise AssertionError(message)

        # Four batched lanes share a per-step threshold and keep elementwise child state.
        #: Encoder that converts the four stacked complex-input components into spikes.
        self.encoder_x = encode(codec_config)
        #: Decoder that tracks the four progressively decoded butterfly outputs.
        self.decoder_y = decode(codec_config)
        #: Accuracy monitor that accumulates the four output spike streams.
        self.accuracy_y = accuracy(acc_config)
        #: Conditional-spike multiplier for the four stacked twiddle products.
        self.mul_wx = mul_ugemm(mul_config)
        #: Scaled unary adder that combines each input with its twiddle term.
        self.add_y = add_any(add_config)

        # Stacks derived from the inputs; buffers so they follow the module device.
        #: Stacked complex input components fed to the encoder.
        self.x_stack: torch.Tensor
        self.register_buffer('x_stack', None, persistent=False)
        #: Stacked twiddle components fed to the multiplier.
        self.w_stack: torch.Tensor
        self.register_buffer('w_stack', None, persistent=False)
        #: Per-lane sign that turns a subtraction into a complemented stream.
        self.sign: torch.Tensor
        self.register_buffer('sign', None, persistent=False)
        #: Bias absorbing the complement constant on the first output lane.
        self.bias0: torch.Tensor
        self.register_buffer('bias0', None, persistent=False)
        #: Bias absorbing the complement constant on the second output lane.
        self.bias1: torch.Tensor
        self.register_buffer('bias1', None, persistent=False)
        #: Leading batch size the cached stacks were built for.
        self.batch = None
        # The cache is valid only while the same input tensors hold the same versions.
        self._stack_inputs = None

        # Encoding, multiplication, and addition are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming butterfly.
        self.hw = hw_params(pp_delay=0)

        #: Empty, since the numeric ports carry no stream encoding.
        self.encoding_io = {}
        #: Value range of all ten numeric ports, which share the class polarity.
        self.polarity_io = {port: self.polarity for port in
                            ('x0r', 'x0i', 'x1r', 'x1i', 'wr', 'wi', 'y0r', 'y0i', 'y1r', 'y1i')}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Clear the class-local cache of tensors derived from the inputs.

        The inherited reset method also resets the timestep and every child NAPL
        module before calling this hook. This hook returns ``None``.
        """
        self.x_stack = None
        self.w_stack = None
        self.sign = None
        self.bias0 = None
        self.bias1 = None
        self.batch = None
        self._stack_inputs = None


    def forward(self, x0r, x0i, x1r, x1i, wr, wi):
        """
        Process one spike timestep of the configured butterfly.

        The six tensors must be broadcast-compatible and have at least one
        dimension. Call the module once per timestep of the run.

        Args:
            x0r: Real part of the first complex input.
            x0i: Imaginary part of the first complex input.
            x1r: Real part of the second complex input.
            x1i: Imaginary part of the second complex input.
            wr: Real part of the twiddle factor.
            wi: Imaginary part of the twiddle factor.

        Returns:
            A tuple ``(y0r, y0i, y1r, y1i)`` of progressively decoded tensors,
            each with the broadcast input shape.

        The call advances the child encoder, multiplier, adder, decoder, and
        accuracy metric, and increments this module's timestep once. The inputs
        themselves are not modified.

        **Example:**

        .. code-block:: python

            operation.reset()
            for _ in range(4):
                outputs = operation(*x0, *x1, *w)
        """
        x_stack, w_stack, sign, bias0, bias1, b = self._stacks(x0r, x0i, x1r, x1i, wr, wi)

        x_spike = self.encoder_x(x_stack)
        x0_spike = x_spike.narrow(0, 0, 2 * b)
        x1_spike = x_spike.narrow(0, 2 * b, 2 * b)

        m = self.mul_wx(torch.cat([x1_spike, x1_spike], 0), w_stack)
        m01 = m.narrow(0, 0, 2 * b)
        m32 = torch.cat([m.narrow(0, 3 * b, b), m.narrow(0, 2 * b, b)], 0)

        # Bias terms absorb inverted spikes in the real-minus and imaginary-plus lanes.
        t = m01 + sign * m32
        y0_sum = x0_spike + t + bias0
        y1_sum = x0_spike - t + bias1

        y_spike = self.add_y(torch.cat([y0_sum, y1_sum], 0), entry=3, dim=None)
        self.decoder_y(y_spike)
        y = self.decoder_y.spike_value
        self.accuracy_y(y_spike)

        return y.narrow(0, 0, b), y.narrow(0, b, b), y.narrow(0, 2 * b, b), y.narrow(0, 3 * b, b)


    def _stacks(self, x0r, x0i, x1r, x1i, wr, wi):
        """Build, or reuse, the stacked operands and per-lane constants.

        The cached stacks are returned while the same input tensor objects carry
        the same versions, and are rebuilt otherwise. The cache holds references
        to the inputs and compares them by object identity.
        """
        inputs = (x0r, x0i, x1r, x1i, wr, wi)
        if self._stack_inputs is not None:
            cached, versions = self._stack_inputs
            if all(a is b for a, b in zip(cached, inputs)) \
                    and all(t._version == v for t, v in zip(inputs, versions)):
                return self.x_stack, self.w_stack, self.sign, self.bias0, self.bias1, self.batch
        shape = torch.broadcast_shapes(x0r.shape, x0i.shape, x1r.shape, x1i.shape, wr.shape, wi.shape)
        x0r, x0i, x1r, x1i, wr, wi = (t.expand(shape) for t in inputs)
        self.x_stack = torch.cat([x0r, x0i, x1r, x1i], 0)
        self.w_stack = torch.cat([wr, wr, wi, wi], 0)
        b = shape[0]
        tail = (1,) * (len(shape) - 1)
        self.sign = torch.cat([torch.full((b,) + tail, -1),
                               torch.full((b,) + tail, 1)]).type(self.stype).to(self.x_stack.device)
        self.bias0 = self.sign.eq(-1).type(self.stype)
        self.bias1 = torch.cat([self.bias0.narrow(0, 0, b), self.bias0.narrow(0, 0, b) + 1])
        self.batch = b
        self._stack_inputs = (inputs, tuple(t._version for t in inputs))
        return self.x_stack, self.w_stack, self.sign, self.bias0, self.bias1, self.batch

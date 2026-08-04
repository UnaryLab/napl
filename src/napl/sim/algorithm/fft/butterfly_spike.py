import torch

from napl.sim.base import napl_base, napl_sim_timesteps
from napl.sim.module import encoder, decoder
from napl.sim.metric import accuracy
from napl.sim.operation import mul_csg, add_any


class butterfly_spike(napl_base):
    """
    Evaluate a radix-2 complex butterfly through streaming spike operations.

    Use this class for a progressively decoded unary FFT stage. Use
    :class:`butterfly_binary` when an exact binary-domain reference is needed.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.algorithm.fft import butterfly_spike

        codec = {'polarity': 'bipolar', 'timestep': 4, 'generator': 'sobol'}
        adder = {'polarity': 'bipolar', 'scale': 3, 'width': 3}
        operation = butterfly_spike(codec, codec, adder, {'polarity': 'bipolar'})
        inputs = tuple(torch.zeros(1) for _ in range(6))
        y0r, y0i, y1r, y1i = operation(*inputs, timesteps=4)
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

              - **polarity**: Required stream encoding, either ``"unipolar"``
                or ``"bipolar"``.
              - **timestep**: Required positive maximum decoder run length.
              - **generator**: Required encoder number-sequence generator. The
                accepted values are ``"sobol"``, ``"lfsr"``, ``"sys"``,
                ``"rc"``, ``"tc"``, ``"rate"``, and ``"temporal"``.
              - **dim**: Sobol dimension; the default is ``1``.
              - **seed**: LFSR seed; the default is ``None``.
              - **taps**: LFSR feedback taps; the default is ``None``.
              - **name**: Optional component label; the default is ``None``.

            - **mul_config** – Conditional-spike multiplier configuration.

              - **polarity**: Required stream encoding, either ``"unipolar"``
                or ``"bipolar"``.
              - **timestep**: Required positive sequence length basis.
              - **generator**: Required number-sequence generator, with the same
                accepted values as ``codec_config``.
              - **name**: Optional component label; the default is ``None``.

            - **add_config** – Scaled-adder configuration.

              - **polarity**: Required stream encoding, either ``"unipolar"``
                or ``"bipolar"``.
              - **scale**: Required output carry scale.
              - **width**: Required signed accumulator width in bits.
              - **name**: Optional component label; the default is ``None``.

            - **acc_config** – Output accuracy-metric configuration.

              - **polarity**: Required stream encoding, either ``"unipolar"``
                or ``"bipolar"``.
              - **name**: Optional metric label; the default is ``None``.
        """
        super().__init__()

        # Four batched lanes share a per-step threshold and keep elementwise child state.
        #: Encoder that converts the four stacked complex-input components into spikes.
        self.encoder_x = encoder(codec_config)
        #: Decoder that tracks the four progressively decoded butterfly outputs.
        self.decoder_y = decoder(codec_config)
        #: Accuracy monitor that accumulates the four output spike streams.
        self.accuracy_y = accuracy(acc_config)
        #: Conditional-spike multiplier for the four stacked twiddle products.
        self.mul_wx = mul_csg(mul_config)
        #: Scaled unary adder that combines each input with its twiddle term.
        self.add_y = add_any(add_config)

        # Cached stacks are valid only for the same tensor identity, version, and shape.
        self._stack_cache = None


    def _reset(self):
        """
        Clear the class-local cache of tensors derived from the inputs.

        The inherited reset method also resets the timestep and every child NAPL
        module before calling this hook. This hook returns ``None``.
        """
        self._stack_cache = None


    @napl_sim_timesteps
    def forward(self, x0r, x0i, x1r, x1i, wr, wi):
        """
        Simulate the configured butterfly for multiple spike timesteps.

        The six tensors must be broadcast-compatible and have at least one
        dimension. Call the module with the required keyword-only simulation
        control ``timesteps``.

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

        Each internal timestep advances the child encoder, multiplier, adder,
        decoder, and accuracy metric. One outer module call increments this
        module's timestep once. The inputs themselves are not modified.

        **Example:**

        .. code-block:: python

            outputs = operation(*inputs, timesteps=4)
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
        key = tuple((t.data_ptr(), t._version, t.shape) for t in (x0r, x0i, x1r, x1i, wr, wi))
        if self._stack_cache is not None and self._stack_cache[0] == key:
            return self._stack_cache[1:]
        shape = torch.broadcast_shapes(x0r.shape, x0i.shape, x1r.shape, x1i.shape, wr.shape, wi.shape)
        x0r, x0i, x1r, x1i, wr, wi = (t.expand(shape) for t in (x0r, x0i, x1r, x1i, wr, wi))
        x_stack = torch.cat([x0r, x0i, x1r, x1i], 0)
        w_stack = torch.cat([wr, wr, wi, wi], 0)
        b = shape[0]
        tail = (1,) * (len(shape) - 1)
        sign = torch.cat([torch.full((b,) + tail, -1), torch.full((b,) + tail, 1)]).type(self.stype).to(x_stack.device)
        bias0 = sign.eq(-1).type(self.stype)
        bias1 = torch.cat([bias0.narrow(0, 0, b), bias0.narrow(0, 0, b) + 1])
        self._stack_cache = (key, x_stack, w_stack, sign, bias0, bias1, b)
        return x_stack, w_stack, sign, bias0, bias1, b

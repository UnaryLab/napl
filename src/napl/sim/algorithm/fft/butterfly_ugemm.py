import torch
from loguru import logger

from napl.sim.base import napl_base
from napl.sim.operation import add_scale, mul_ugemm


class butterfly_ugemm(napl_base):
    r"""
    Evaluate a radix-2 complex butterfly with spike streams on every data port.

    Use this class to place a butterfly inside a fully streaming pipeline: it
    takes bipolar 0/1 spike tensors and returns bipolar 0/1 spike tensors, one
    timestep per call. It holds no encoder and no decoder, so a caller encodes
    upstream and decodes downstream, and a chain of these classes never leaves
    the spike domain. This is the difference from :class:`fft_hub`, whose ports
    carry numeric values because it encodes and progressively decodes at the
    boundary of the transform it wraps, and from :class:`butterfly_fp`, which is
    an exact binary-domain reference.

    The target is the radix-2 decimation-in-time butterfly on complex inputs
    :math:`x_0`, :math:`x_1` and twiddle factor :math:`w`,

    .. math::

       y_0 = x_0 + w x_1,\qquad y_1 = x_0 - w x_1.

    The twiddle factor is a constant of the stage, so it is supplied at
    construction and turned into a conditional spike stream inside the
    multiplier rather than arriving on a port. A stream cannot represent a
    magnitude above one, so the returned streams encode the target divided by
    the adder scale reported in :attr:`compensation`,

    .. math::

       \hat y_0 \approx \frac{x_0 + w x_1}{\mathit{compensation}},\qquad
       \hat y_1 \approx \frac{x_0 - w x_1}{\mathit{compensation}}.

    The class does not undo that division; a consumer that wants the unscaled
    butterfly multiplies the decoded stream value by :attr:`compensation`.

    Only bipolar encoding is supported, because :math:`y_1 = x_0 - w x_1` is
    negative for positive operands and a unipolar stream cannot represent that.

    The multiplier holds its own number sequence and accepts no sequence
    dimension, so every instance built from the same ``mul_config`` carries a
    bit-identical weight sequence. Chained instances still decorrelate, because
    ``mul_ugemm`` advances its sequence indices conditionally on the data: fed
    distinct data, their index pointers diverge and they read different points
    of the shared sequence.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import decode, encode
        from napl.sim.algorithm import butterfly_ugemm

        codec = {'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol'}
        adder = {'polarity': 'bipolar', 'scale': 3, 'intwidth': 6, 'fracwidth': 0}
        twiddle_real = torch.tensor([0.5])
        twiddle_imag = torch.tensor([0.5])
        operation = butterfly_ugemm(twiddle_real, twiddle_imag, codec, adder)
        source = encode(codec)
        sink = decode(codec)
        values = torch.tensor([[0.5], [0.25], [0.5], [-0.25]])
        for _ in range(256):
            spikes = source(values)
            outputs = operation(*spikes.split(1, dim=0))
            sink(outputs[0])
        print(round((sink.spike_value * operation.compensation).item(), 4))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """


    def __init__(
            self,
            twiddle_real,
            twiddle_imag,
            mul_config,
            add_config,
        ):
        """
        Configure the constant twiddle factor, the multiplier, and the adder.

        .. container:: api-parameter-list

            **Parameters:**

            - **twiddle_real** – Non-empty 1-D tensor holding the real part of
              the constant twiddle factor, one entry per butterfly lane. Its
              length fixes the leading dimension of every spike port.

            - **twiddle_imag** – Imaginary part of the same twiddle factor, with
              the same length as ``twiddle_real``.

            - **mul_config** – Conditional-spike multiplier configuration.

              - **polarity**: Required stream encoding, which must be
                ``"bipolar"``.
              - **timestep**: Required positive sequence length basis.
              - **generator**: Required number-sequence generator. The accepted
                values are ``"sobol"``, ``"lfsr"``, ``"sys"``, ``"rc"``,
                ``"tc"``, ``"rate"``, and ``"temporal"``.
              - **name**: Optional component label; the default is ``None``.

            - **add_config** – Scaled-adder configuration.

              - **polarity**: Required stream encoding, which must be
                ``"bipolar"``.
              - **scale**: Required output carry scale, reported afterwards as
                :attr:`compensation`.
              - **intwidth**: Required integer bits of the signed accumulator,
                including the sign bit.
              - **fracwidth**: Required fractional bits of the signed
                accumulator. The accumulator rail
                ``acc_max = 2 ** (intwidth + fracwidth - 1) - 1`` must satisfy
                ``acc_max >= (scale_raw - grid) + delta_max`` in raw units of
                ``2 ** -fracwidth``, where the adder fan-in is
                ``entry_raw = 3 * 2 ** fracwidth``, ``scale_raw`` is the
                quantized scale in raw units, ``delta_max`` is the largest
                per-timestep accumulator step ``(entry_raw + scale_raw) / 2``
                for this bipolar adder, and ``grid`` is the accumulator step
                (``0.5`` when ``entry_raw - scale_raw`` is odd, else ``1``).
                This bound is static for ``scale >= entry``; for
                ``scale < entry`` the accumulator must also satisfy
                ``acc_max + 1 > entry_raw``, a minimum burst-headroom floor
                rather than a safety bound, since the accumulator then drains by
                at most ``scale`` per timestep and correctness is conditional on
                the long-run mean inflow staying below ``scale`` (see
                :class:`add_scale`).
              - **name**: Optional component label; the default is ``None``.

        Construction stores the twiddle factor and the per-lane sign and bias
        constants, so no configuration mapping is modified.
        """
        super().__init__(mul_config, ['polarity', 'timestep', 'generator'],
                         polarity_required=True)
        # y1 = x0 - w x1 is negative for positive operands, which a unipolar stream
        # cannot represent, so the subtraction path requires bipolar encoding.
        if self.polarity != 'bipolar':
            message = f'Invalid polarity: <{self.polarity}>; legal values: <[\'bipolar\']>.'
            logger.error(message)
            raise AssertionError(message)

        for label, value in (('twiddle_real', twiddle_real), ('twiddle_imag', twiddle_imag)):
            if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() == 0:
                shape = tuple(value.shape) if isinstance(value, torch.Tensor) \
                    else type(value).__name__
                message = (
                    f'Invalid {label}: <{shape}>; legal values: a non-empty 1-D tensor.'
                )
                logger.error(message)
                raise AssertionError(message)
        if twiddle_real.numel() != twiddle_imag.numel():
            message = (
                f'butterfly_ugemm twiddle_real length <{twiddle_real.numel()}> must '
                f'equal twiddle_imag length <{twiddle_imag.numel()}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Number of independent butterfly lanes carried by the leading dimension.
        self.lane = twiddle_real.numel()
        #: Conditional-spike multiplier for the four stacked twiddle products.
        self.mul_wx = mul_ugemm(dict(mul_config))
        #: Adder fan-in covering the input spike and the two twiddle-product spikes.
        self.add_entry = 3
        #: Scaled unary adder that combines each input spike with its twiddle term.
        self.add_y = add_scale(dict(add_config))

        adder_scale = self.add_y.scale
        # The fan-in and step bounds are restated in the accumulator's raw units of
        # 2 ** -fracwidth to check that its rail holds the largest sub-threshold residue
        # (scale - grid) plus one timestep's step delta_max, plus one worst-case burst
        # timestep when scale is below the fan-in.
        acc_max = self.add_y.acc_max
        entry_raw = self.add_entry * 2**self.add_y.fracwidth
        scale_raw = self.add_y.scale_raw
        delta_max = (entry_raw + scale_raw) / 2
        grid = 0.5 if (entry_raw - scale_raw) % 2 else 1
        if (acc_max < (scale_raw - grid) + delta_max
                or (scale_raw < entry_raw and acc_max + 1 <= entry_raw)):
            message = (
                f'butterfly_ugemm accumulator maximum <{acc_max}> raw units too small for fan-in '
                f'<{self.add_entry}> and scale <{adder_scale}>: for this bipolar-only adder '
                f'acc_max must be >= (scale_raw - grid) + delta_max, with grid <{grid}> '
                f'and delta_max <{delta_max}> in raw units of <{self.add_y.grid}>, and acc_max + 1 '
                f'must be > entry when scale < entry, or partial sums saturate. Increase intwidth.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Factor the output streams are divided by, equal to the adder scale.
        self.compensation = adder_scale

        # The twiddle is a stage constant, so its stacked operand and the per-lane
        # sign and bias constants are built once instead of per call.
        #: Stacked twiddle components read by the multiplier on every timestep.
        self.twiddle_stack: torch.Tensor
        self.register_buffer(
            'twiddle_stack',
            torch.cat([twiddle_real, twiddle_real, twiddle_imag, twiddle_imag])
            .detach().type(self.ntype),
        )
        sign = torch.cat([
            torch.full((self.lane,), -1),
            torch.full((self.lane,), 1),
        ]).type(self.stype)
        #: Per-lane sign that turns a subtraction into a complemented stream.
        self.sign: torch.Tensor
        self.register_buffer('sign', sign)
        #: Bias absorbing the complement constant on the first output lane.
        self.bias0: torch.Tensor
        self.register_buffer('bias0', sign.eq(-1).type(self.stype))
        #: Bias absorbing the complement constant on the second output lane.
        self.bias1: torch.Tensor
        self.register_buffer(
            'bias1',
            torch.cat([self.bias0.narrow(0, 0, self.lane),
                       self.bias0.narrow(0, 0, self.lane) + 1]),
        )

        # Multiplication and addition are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming butterfly.
        self.hw.pp_delay = 0
        #: Whether the RTL counterpart must hold its own encoder, true when any part does.
        self.internal_encode = any(part.internal_encode for part in self.children())

        #: Rate coding on every spike port, which the class neither encodes nor decodes.
        self.encoding_io = {port: 'rc' for port in
                            ('x0r', 'x0i', 'x1r', 'x1i', 'y0r', 'y0i', 'y1r', 'y1i')}
        #: Stream polarity of all eight spike ports, which share the class polarity.
        self.polarity_io = {port: self.polarity for port in
                            ('x0r', 'x0i', 'x1r', 'x1i', 'y0r', 'y0i', 'y1r', 'y1i')}
        self.correlation_i = {}


    def _reset(self):
        """
        Reset class-local execution state.

        The twiddle, sign, and bias constants are fixed at construction, so this
        class holds no local mutable state. The inherited reset method resets the
        timestep, the multiplier, and the adder before this hook returns ``None``.
        """
        pass


    def forward(self, x0r, x0i, x1r, x1i):
        """
        Process one spike timestep of the configured butterfly.

        Args:
            x0r: Real-part spike tensor of the first complex input, shaped
                ``(lane, ...)``.
            x0i: Imaginary-part spike tensor of the first complex input, with the
                same shape as ``x0r``.
            x1r: Real-part spike tensor of the second complex input, with the
                same shape as ``x0r``.
            x1i: Imaginary-part spike tensor of the second complex input, with
                the same shape as ``x0r``.

        Returns:
            A tuple ``(y0r, y0i, y1r, y1i)`` of bipolar 0/1 spike tensors with
            the input shape, each encoding its exact butterfly output divided by
            :attr:`compensation`.

        The call advances the multiplier and the adder once and increments this
        module's timestep. The inputs are not modified.

        **Example:**

        .. code-block:: python

            operation.reset()
            for _ in range(4):
                y0r, y0i, y1r, y1i = operation(*spikes)
        """
        if not (x0r.shape == x0i.shape == x1r.shape == x1i.shape):
            message = (
                f'butterfly_ugemm input shapes must match: got <{x0r.shape}>, '
                f'<{x0i.shape}>, <{x1r.shape}>, and <{x1i.shape}>.'
            )
            logger.error(message)
            raise AssertionError(message)
        if x0r.ndim == 0 or x0r.shape[0] != self.lane:
            message = (
                f'butterfly_ugemm first input dimension must equal lane <{self.lane}>: '
                f'got shape <{x0r.shape}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        lane = self.lane
        tail = (1,) * (x0r.ndim - 1)
        x0_spike = torch.cat([x0r, x0i], 0)
        x1_spike = torch.cat([x1r, x1i], 0)

        product = self.mul_wx(
            torch.cat([x1_spike, x1_spike], 0),
            self.twiddle_stack.reshape((-1,) + tail),
        )
        product_01 = product.narrow(0, 0, 2 * lane)
        product_32 = torch.cat([
            product.narrow(0, 3 * lane, lane),
            product.narrow(0, 2 * lane, lane),
        ], 0)

        # Bias terms absorb inverted spikes in the real-minus and imaginary-plus lanes.
        term = product_01 + self.sign.reshape((-1,) + tail) * product_32
        y0_sum = x0_spike + term + self.bias0.reshape((-1,) + tail)
        y1_sum = x0_spike - term + self.bias1.reshape((-1,) + tail)

        y_spike = self.add_y(
            torch.cat([y0_sum, y1_sum], 0), entry=self.add_entry, dim=None
        )
        return (
            y_spike.narrow(0, 0, lane),
            y_spike.narrow(0, lane, lane),
            y_spike.narrow(0, 2 * lane, lane),
            y_spike.narrow(0, 3 * lane, lane),
        )

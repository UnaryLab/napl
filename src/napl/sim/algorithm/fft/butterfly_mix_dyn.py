import torch
from loguru import logger

from napl.sim.base import napl_base
from napl.sim.operation import add_scale, add_scale_dyn, encode, mul_gaines


class butterfly_mix_dyn(napl_base):
    r"""Evaluate a Gaines radix-2 butterfly with a runtime adder scale.

    Use this class when a butterfly must stay in the spike domain on every data
    port, the twiddle product is an XNOR gate, and the carry scale is chosen on
    each timestep. Like :class:`butterfly_mix`, it takes bipolar 0/1 spike
    tensors and returns bipolar 0/1 spike tensors and holds no encoder or
    decoder on the data ports, so it is not the numeric-port class
    :class:`fft_dyn_hub`.

    With the scale held constant from reset through a run, the output streams
    encode the exact butterfly divided by that scale, reported after each call
    as :attr:`compensation`. ``add_scale_dyn`` preserves accumulator mass at the
    scale active when a spike fires, so a scale may change without reset, but
    the stream then mixes segments produced under different scales and no single
    compensation factor describes the whole run. Only bipolar encoding is
    supported.

    As in :class:`butterfly_mix`, the twiddle factor is encoded inside the class
    on its own Sobol dimension, ``dim``, which must differ from the dimensions
    feeding the four input ports, because an XNOR multiplier is exact only for
    uncorrelated operands.

    .. container:: api-references

        .. rubric:: References

        *Stochastic Computing Systems*, Advances in Information Systems Science, 1969.
    """
    #: The twiddle encoder is held inside the class, so the RTL counterpart
    #: encodes that operand itself from a held numeric code instead of taking it
    #: as a spike from a shared encoder.
    internal_encode = True


    def __init__(self, twiddle_real, twiddle_imag, mul_config, add_config):
        """Configure the constant twiddle factor, the multiplier, and the dynamic adder.

        Args:
            twiddle_real: Non-empty 1-D tensor holding the real part of the
                constant twiddle factor, one entry per butterfly lane.
            twiddle_imag: Imaginary part of the same twiddle factor, with the
                same length as ``twiddle_real``.
            mul_config: Bipolar Gaines multiplier and twiddle-encoder
                configuration containing ``polarity``, ``timestep``, and
                ``generator``, plus the optional one-based Sobol dimension
                ``dim`` of the twiddle encoder, which defaults to ``5``. Only
                the sobol-family generators ``"sobol"``, ``"rc"``, and
                ``"rate"`` honor ``dim``; the other accepted values give the
                twiddle encoder the same sequence as the input encoders, which
                correlates the multiplier operands and logs a warning.
            add_config: Dynamic adder configuration containing ``polarity``,
                positive ``scale_max``, ``intwidth``, and ``fracwidth``. The
                accumulator bound documented on :class:`butterfly_mix` is
                checked against ``scale_max``.
        """
        if 'scale_max' not in add_config:
            message = 'Missing key <scale_max> in the dynamic adder configuration.'
            logger.error(message)
            raise AssertionError(message)
        scale_max = add_config['scale_max']
        if type(scale_max) is not int or scale_max < 1:
            message = (
                f'butterfly_mix_dyn scale_max must be a positive int: '
                f'got <{scale_max}>.'
            )
            logger.error(message)
            raise AssertionError(message)
        static_add_config = dict(add_config)
        static_add_config['scale'] = static_add_config.pop('scale_max')
        super().__init__(mul_config, ['polarity', 'timestep', 'generator'],
                         optional_key_list=['dim'], polarity_required=True)
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
                f'butterfly_mix twiddle_real length <{twiddle_real.numel()}> must '
                f'equal twiddle_imag length <{twiddle_imag.numel()}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Number of independent butterfly lanes carried by the leading dimension.
        self.lane = twiddle_real.numel()
        #: Gaines XNOR multiplier for the four stacked twiddle products.
        self.mul_wx = mul_gaines({'polarity': self.polarity,
                                  'name': mul_config.get('name')})
        # The XNOR multiplier needs the twiddle as a stream, and only a Sobol-family
        # generator decorrelates it from x1 by dimension, so the default dimension 5 sits
        # above the dimensions 1 to 4 the four input ports conventionally use.
        if mul_config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'butterfly_mix_dyn decorrelates operands via distinct sobol dimensions, but '
                f'generator <{mul_config["generator"]}> does not decorrelate by dim (identical '
                f'sequences across operands). Use a sobol-family generator, or decorrelate the '
                f'input and twiddle streams by distinct seeds.')
        #: Encoder turning the stacked twiddle constants into a spike stream.
        self.reference_encode = encode({
            'polarity': self.polarity,
            'timestep': mul_config['timestep'],
            'generator': mul_config['generator'],
            'dim': mul_config.get('dim', 5),
            'name': mul_config.get('name'),
        })
        #: Adder fan-in covering the input spike and the two twiddle-product spikes.
        self.add_entry = 3

        # The static adder is a construction-time check that the largest runtime
        # scale fits the requested accumulator width.
        static_add = add_scale(static_add_config)
        adder_scale = static_add.scale
        # The fan-in and step bounds are restated in the accumulator's raw units of
        # 2 ** -fracwidth to check that its rail holds the largest sub-threshold residue
        # (scale - grid) plus one timestep's step delta_max, plus one worst-case burst
        # timestep when scale is below the fan-in.
        acc_max = static_add.acc_max
        entry_raw = self.add_entry * 2**static_add.fracwidth
        scale_raw = static_add.scale_raw
        delta_max = (entry_raw + scale_raw) / 2
        grid = 0.5 if (entry_raw - scale_raw) % 2 else 1
        if (acc_max < (scale_raw - grid) + delta_max
                or (scale_raw < entry_raw and acc_max + 1 <= entry_raw)):
            message = (
                f'butterfly_mix accumulator maximum <{acc_max}> raw units too small for fan-in '
                f'<{self.add_entry}> and scale <{adder_scale}>: for this bipolar-only adder '
                f'acc_max must be >= (scale_raw - grid) + delta_max, with grid <{grid}> '
                f'and delta_max <{delta_max}> in raw units of <{static_add.grid}>, and acc_max + 1 '
                f'must be > entry when scale < entry, or partial sums saturate. Increase intwidth.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Dynamic scaled adder used by the butterfly output paths.
        self.add_y = add_scale_dyn(dict(add_config))
        #: Largest runtime carry scale accepted by this butterfly.
        self.scale_max = self.add_y.scale_max
        #: Factor the output streams of the most recent call are divided by.
        self.compensation = None

        # The twiddle is a stage constant, so its stacked operand and the per-lane
        # sign and bias constants are built once instead of per call.
        #: Stacked twiddle components encoded on every timestep.
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

        # Encoding, multiplication, and addition are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming butterfly.
        self.hw.pp_delay = 0

        #: Rate coding on every spike port, which the class neither encodes nor decodes.
        self.encoding_io = {port: 'rc' for port in
                            ('x0r', 'x0i', 'x1r', 'x1i', 'y0r', 'y0i', 'y1r', 'y1i')}
        #: Stream polarity of all eight spike ports, which share the class polarity.
        self.polarity_io = {port: self.polarity for port in
                            ('x0r', 'x0i', 'x1r', 'x1i', 'y0r', 'y0i', 'y1r', 'y1i')}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Clear the reported compensation after the registered children reset.

        The twiddle, sign, and bias constants are fixed at construction, so the
        reported factor is the only class-local state. It returns to ``None``
        because no call has selected a scale yet. This hook returns ``None``.
        """
        self.compensation = None


    def forward(self, x0r, x0i, x1r, x1i, scale):
        """Process one butterfly timestep at the requested carry scale.

        Args:
            x0r: Real-part spike tensor of the first complex input, shaped
                ``(lane, ...)``.
            x0i: Imaginary-part spike tensor of the first complex input.
            x1r: Real-part spike tensor of the second complex input.
            x1i: Imaginary-part spike tensor of the second complex input.
            scale: Integer carry scale in ``1`` through ``scale_max`` for this
                timestep. This control value is not a spike port.

        Returns:
            ``(y0r, y0i, y1r, y1i)`` as bipolar 0/1 spike tensors, each encoding
            its exact butterfly output divided by :attr:`compensation`.

        The call advances every child once and updates the dynamic adder
        accumulator at ``scale`` without modifying the input spikes. Hold
        ``scale`` constant after reset so that one compensation factor describes
        the whole stream.
        """
        scale = self._runtime_scale(scale)
        if not (x0r.shape == x0i.shape == x1r.shape == x1i.shape):
            message = (
                f'butterfly_mix_dyn input shapes must match: got <{x0r.shape}>, '
                f'<{x0i.shape}>, <{x1r.shape}>, and <{x1i.shape}>.'
            )
            logger.error(message)
            raise AssertionError(message)
        if x0r.ndim == 0 or x0r.shape[0] != self.lane:
            message = (
                f'butterfly_mix_dyn first input dimension must equal lane <{self.lane}>: '
                f'got shape <{x0r.shape}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        lane = self.lane
        tail = (1,) * (x0r.ndim - 1)
        x0_spike = torch.cat([x0r, x0i], 0)
        x1_spike = torch.cat([x1r, x1i], 0)

        reference_encode_bit = self.reference_encode(
            self.twiddle_stack.reshape((-1,) + tail)
        )
        product = self.mul_wx(
            torch.cat([x1_spike, x1_spike], 0),
            reference_encode_bit,
        )
        product_01 = product.narrow(0, 0, 2 * lane)
        product_32 = torch.cat([
            product.narrow(0, 3 * lane, lane),
            product.narrow(0, 2 * lane, lane),
        ], 0)

        term = product_01 + self.sign.reshape((-1,) + tail) * product_32
        y0_sum = x0_spike + term + self.bias0.reshape((-1,) + tail)
        y1_sum = x0_spike - term + self.bias1.reshape((-1,) + tail)

        y_spike = self.add_y(
            torch.cat([y0_sum, y1_sum], 0), scale, entry=self.add_entry, dim=None
        )
        self.compensation = scale
        return (
            y_spike.narrow(0, 0, lane),
            y_spike.narrow(0, lane, lane),
            y_spike.narrow(0, 2 * lane, lane),
            y_spike.narrow(0, 3 * lane, lane),
        )


    def __call__(self, x0r, x0i, x1r, x1i, scale):
        """Validate the runtime scale, then process one butterfly timestep.

        Args:
            x0r: Real-part spike tensor of the first complex input.
            x0i: Imaginary-part spike tensor of the first complex input.
            x1r: Real-part spike tensor of the second complex input.
            x1i: Imaginary-part spike tensor of the second complex input.
            scale: Exact Python int in ``1`` through ``scale_max``.

        Returns:
            ``(y0r, y0i, y1r, y1i)`` as bipolar 0/1 spike tensors.

        An invalid scale raises before this module or any child advances or
        updates persistent state. A valid call advances each exactly once.
        """
        self._runtime_scale(scale)
        return super().__call__(x0r, x0i, x1r, x1i, scale)


    def _runtime_scale(self, scale):
        """Return a validated runtime carry scale."""
        if type(scale) is not int:
            message = f'butterfly_mix_dyn scale must be an int: got <{scale}>.'
            logger.error(message)
            raise AssertionError(message)
        if scale < 1 or scale > self.add_y.scale_max:
            message = (
                f'butterfly_mix_dyn scale <{scale}> outside the supported '
                f'range <1> to scale_max <{self.add_y.scale_max}>.'
            )
            logger.error(message)
            raise AssertionError(message)
        return scale

from loguru import logger

from napl.sim.base import napl_base, napl_sim_timesteps
from napl.sim.operation import decode, encode

from .fft import TWIDDLE_DIM_FIRST
from .fft_dyn import fft_dyn


class fft_dyn_hub(napl_base):
    r"""Evaluate a radix-2 FFT with numeric ports and one runtime scale per stage.

    Use this class when the caller works in the binary domain and the per-stage
    carry scales are chosen while the stream runs. Like :class:`fft_hub`, it
    encodes only at its two input ports and decodes only at its two output
    ports, and the transform it wraps is :class:`fft_dyn`, reachable as
    ``operation.core``.

    A scale list held constant from reset approximates the analytic FFT when at
    most one stage uses scale 1. Lists with two or more scale-1 stages remain
    legal but can clip because a scale-1 adder drains at most one unit per
    timestep against a fan-in of three. The list passed to a call holds the
    requested per-stage scales, and the first stage's internal adder runs at
    twice the requested ``scales[0]``, so that value must not exceed half of
    ``scale_max``. This class exposes no ``scales`` attribute, because its
    scales arrive as call inputs.

    After a scale change without reset, the progressive output mixes decoder
    history accumulated under the prior compensation, so decode a run whose
    scales never changed.
    """
    #: Whether each call is one streaming timestep; one hub call is a whole run.
    streaming = False


    def __init__(self, point, codec_config, mul_config, add_config):
        """Configure the boundary codecs and the dynamic transform they wrap.

        Args:
            point: Positive power-of-two transform length.
            codec_config: Shared bipolar encoder and decoder configuration. Its
                **dim** is the real-part encoder's number-sequence dimension and
                **dim** + 1 is the imaginary-part encoder's, so the two input
                streams decorrelate; the default **dim** is ``1``.
            mul_config: Bipolar multiplier configuration. Its optional
                ``kernel`` passes through to the wrapped transform and selects
                the stage class: ``'ugemm'``, the default, or ``'mix'``. With
                ``'mix'``, stage ``index`` encodes its twiddle on Sobol
                dimension ``5 + index``, and both input encoder dimensions
                **dim** and **dim** + 1 must stay outside that range.
            add_config: Dynamic adder configuration containing ``polarity``,
                positive ``scale_max``, ``intwidth``, and ``fracwidth``.

        The three mappings are not modified.
        """
        super().__init__(
            codec_config,
            ['polarity', 'timestep', 'generator'],
            optional_key_list=['dim', 'seed', 'taps'],
            polarity_required=True,
        )
        if self.polarity != 'bipolar':
            message = f'Invalid polarity: <{self.polarity}>; legal values: <[\'bipolar\']>.'
            logger.error(message)
            raise AssertionError(message)

        #: Number of timesteps in one complete run of this transform.
        self.timestep = codec_config['timestep']
        #: Number-sequence dimension used by the real-part input encoder.
        self.dim = codec_config.get('dim', 1)

        #: Input-port encoder turning the real samples into the core's stream.
        self.encode_real = encode(dict(codec_config, dim=self.dim))
        #: Input-port encoder turning the imaginary samples into the core's stream.
        self.encode_imag = encode(dict(codec_config, dim=self.dim + 1))
        #: Streaming spike-domain FFT that owns every butterfly stage.
        self.core = fft_dyn(point, dict(mul_config), dict(add_config))
        #: Output-port decoder holding the progressively decoded real spectrum.
        self.decode_real = decode(dict(codec_config))
        #: Output-port decoder holding the progressively decoded imaginary spectrum.
        self.decode_imag = decode(dict(codec_config))

        #: Number of samples in each transform.
        self.point = self.core.point
        #: Number of radix-2 stages in the transform.
        self.stages = self.core.stages
        #: Largest runtime carry scale accepted by every FFT stage.
        self.scale_max = self.core.scale_max

        # A mix-kernel stage encodes its twiddle on dimension TWIDDLE_DIM_FIRST plus
        # its stage index, so an input encoder on that dimension hands the XNOR
        # multiplier two identical sequences.
        if str(mul_config.get('kernel', 'ugemm')).lower() == 'mix':
            twiddle_dims = range(TWIDDLE_DIM_FIRST, TWIDDLE_DIM_FIRST + self.stages)
            if self.dim in twiddle_dims or self.dim + 1 in twiddle_dims:
                message = (
                    f'Invalid dim: <{self.dim}>; legal values: input encoder dimensions '
                    f'<{self.dim}> and <{self.dim + 1}> must both stay outside the '
                    f'mix-kernel twiddle dimensions <{twiddle_dims.start}> through '
                    f'<{twiddle_dims.stop - 1}>.'
                )
                logger.error(message)
                raise AssertionError(message)

        # Encoding, the core, and decoding are combinational within one timestep.
        #: Hardware latency and timing metadata for the wrapped transform.
        self.hw.pp_delay = self.core.hw.pp_delay
        #: Whether the RTL counterpart must hold its own encoder, true when any part does.
        self.internal_encode = any(part.internal_encode for part in self.children())

        #: Empty, since the numeric ports carry no stream encoding.
        self.encoding_io = {}
        #: Numeric ports use bipolar values inside the streaming transform.
        self.polarity_io = {
            port: self.polarity
            for port in ('input_real', 'input_imag', 'output_real', 'output_imag')
        }
        self.correlation_i = {}


    def _reset(self):
        """Reset state owned directly by this class.

        This class holds no local mutable state beyond the core's reported
        compensation, which the core's own reset clears. This hook returns
        ``None``.
        """
        pass


    def _check_shapes(self, input_real, input_imag):
        """Raise when the numeric input shapes do not match the transform size."""
        if input_real.shape != input_imag.shape:
            message = (
                f'FFT input shapes must match: got <{input_real.shape}> and '
                f'<{input_imag.shape}>.'
            )
            logger.error(message)
            raise AssertionError(message)
        if input_real.ndim == 0 or input_real.shape[0] != self.point:
            message = (
                f'FFT first input dimension must equal point <{self.point}>: '
                f'got shape <{input_real.shape}>.'
            )
            logger.error(message)
            raise AssertionError(message)


    @napl_sim_timesteps
    def forward(self, input_real, input_imag, scales):
        """Run a complete **timestep**-cycle run at the requested runtime stage scales.

        Args:
            input_real: Real values shaped ``(point, ...)`` in natural sample
                order, in ``[-1, 1]``.
            input_imag: Imaginary values with the same shape as ``input_real``.
            scales: One Python int broadcast to all stages, or a Python list
                containing exactly one int per stage. Every value must be in
                ``1`` through ``scale_max``, twice the first value must not
                exceed ``scale_max``, and bool is rejected.

        Returns:
            ``(output_real, output_imag)`` in natural FFT-bin order, with the
            input shape and this call's core compensation applied, taken after
            the final timestep of the run.

        The call is a fresh run: it resets this class and its children, holds
        the numeric input and the scale list fixed for **timestep** timesteps,
        and counts those timesteps on the streaming encoders, core, and
        decoders rather than on this class's ``timestep_cur``. Any state a caller set
        beforehand is discarded, so repeating the call on the same input returns
        the same value. Invalid shapes or scales raise before any child
        advances. Call ``forward_timestep(input_real, input_imag, scales)``
        instead to advance one timestep and read the progressively refined
        spectrum; hold the scale list constant across such a run so that one
        compensation factor describes the whole decoded run.
        """
        self._check_shapes(input_real, input_imag)
        self.core._runtime_scales(scales)
        output_real, output_imag = self.core(
            self.encode_real(input_real), self.encode_imag(input_imag), scales
        )
        self.decode_real(output_real)
        self.decode_imag(output_imag)
        compensation = self.core.compensation
        return (self.decode_real.spike_value * compensation,
                self.decode_imag.spike_value * compensation)

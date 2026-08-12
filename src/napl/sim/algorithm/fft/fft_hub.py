import math

from loguru import logger

from napl.sim.base import napl_base, napl_sim_timesteps
from napl.sim.operation import decode, encode

from .fft import TWIDDLE_DIM_FIRST, fft


class fft_hub(napl_base):
    r"""Evaluate a radix-2 decimation-in-time FFT with numeric input and output ports.

    Use this class when the caller works in the binary domain and wants the
    spike-domain transform to handle the whole stream. Encoding happens only at
    the two input ports and decoding only at the two output ports, so
    :class:`fft` runs purely on spikes in between and no stage owns a codec.

    Each call runs a complete **timestep**-cycle run on a fixed numeric input
    and returns the decoded natural-order spectrum, compensated by
    ``self.core.compensation`` so the returned values match ``torch.fft.fft``
    within the stochastic-computing error of the streams. Inputs are supplied in
    natural order and are reordered inside the core. Stepping the run one
    timestep at a time through ``forward_timestep()`` returns the same spectrum
    progressively refined.

    The first dimension is the transform dimension and must equal ``point``; any
    trailing dimensions are independent transforms processed in parallel. The
    positive integer adder scale may be shared by all stages or supplied as one
    value per stage.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.algorithm import fft_hub

        codec = {'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol'}
        adder = {'polarity': 'bipolar', 'scale': 2, 'intwidth': 8, 'fracwidth': 0}
        operation = fft_hub(8, codec, codec, adder)
        real = torch.zeros(8, 1)
        imag = torch.zeros_like(real)
        spectrum_real, spectrum_imag = operation(real, imag)

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """
    #: Whether each call is one streaming timestep; one hub call is a whole run.
    streaming = False


    def __init__(self, point, codec_config, mul_config, add_config):
        """Configure the boundary codecs and the streaming transform they wrap.

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
            add_config: Bipolar scaled-adder configuration. Its ``scale`` may be
                one positive Python int shared by every stage, or a list
                containing one positive Python int per stage.

        The three mappings are not modified. The wrapped transform is reachable
        as ``operation.core``.
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
        # A mix-kernel stage encodes its twiddle on dimension TWIDDLE_DIM_FIRST plus its
        # stage index, so an input encoder on that dimension would hand the XNOR multiplier
        # two identical sequences; the clash is rejected here for a valid power-of-two point,
        # while an invalid point falls through to the core, which raises the point error.
        point_is_power_of_two = (
            isinstance(point, int) and not isinstance(point, bool)
            and point >= 2 and not (point & (point - 1))
        )
        if str(mul_config.get('kernel', 'ugemm')).lower() == 'mix' and point_is_power_of_two:
            stages = int(math.log2(point))
            twiddle_dims = range(TWIDDLE_DIM_FIRST, TWIDDLE_DIM_FIRST + stages)
            if self.dim in twiddle_dims or self.dim + 1 in twiddle_dims:
                message = (
                    f'Invalid dim: <{self.dim}>; legal values: input encoder dimensions '
                    f'<{self.dim}> and <{self.dim + 1}> must both stay outside the '
                    f'mix-kernel twiddle dimensions <{twiddle_dims.start}> through '
                    f'<{twiddle_dims.stop - 1}>.'
                )
                logger.error(message)
                raise AssertionError(message)

        #: Streaming spike-domain FFT that owns every butterfly stage.
        self.core = self._make_core(point, mul_config, add_config)
        #: Output-port decoder holding the progressively decoded real spectrum.
        self.decode_real = decode(dict(codec_config))
        #: Output-port decoder holding the progressively decoded imaginary spectrum.
        self.decode_imag = decode(dict(codec_config))

        #: Number of samples in each transform.
        self.point = self.core.point
        #: Number of radix-2 stages in the transform.
        self.stages = self.core.stages

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


    def _make_core(self, point, mul_config, add_config):
        """Return the streaming spike-domain transform this class wraps."""
        return fft(point, dict(mul_config), dict(add_config))


    @property
    def scales(self):
        """Requested carry scale per stage; stage 0's adder runs at twice scales[0]."""
        return self.core.scales


    def _reset(self):
        """Reset state owned directly by this class.

        This class holds no local mutable state. The inherited reset method
        restarts the two input encoders, the streaming core, and the two output
        decoders before this hook returns ``None``.
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
    def forward(self, input_real, input_imag):
        """Run a complete **timestep**-cycle run for a fixed numeric input.

        Args:
            input_real: Real values shaped ``(point, ...)`` in natural sample
                order, in ``[-1, 1]``.
            input_imag: Imaginary values with the same shape as ``input_real``.

        Returns:
            ``(output_real, output_imag)`` in natural FFT-bin order, with the
            input shape and the core compensation applied, taken after the final
            timestep of the run.

        The call is a fresh run: it resets this class and its children, holds
        the numeric input fixed for **timestep** timesteps, and counts those
        timesteps on the streaming encoders, core, and decoders rather than on
        this class's ``timestep_cur``. Any state a caller set beforehand is
        discarded, so repeating the call on the same input returns the same
        value. Invalid shapes raise before any child advances. Call
        ``forward_timestep(input_real, input_imag)`` instead to advance one
        timestep and read the progressively refined spectrum.
        """
        self._check_shapes(input_real, input_imag)
        output_real, output_imag = self.core(
            self.encode_real(input_real), self.encode_imag(input_imag)
        )
        self.decode_real(output_real)
        self.decode_imag(output_imag)
        compensation = self.core.compensation
        return (self.decode_real.spike_value * compensation,
                self.decode_imag.spike_value * compensation)

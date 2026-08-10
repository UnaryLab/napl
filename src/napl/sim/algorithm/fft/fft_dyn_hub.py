from .fft_dyn import fft_dyn
from .fft_hub import fft_hub


class fft_dyn_hub(fft_hub):
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


    def __init__(self, point, codec_config, mul_config, add_config):
        """Configure the boundary codecs and the dynamic transform they wrap.

        Args:
            point: Positive power-of-two transform length.
            codec_config: Shared bipolar encoder and decoder configuration.
            mul_config: Bipolar conditional-spike multiplier configuration.
            add_config: Dynamic adder configuration containing ``polarity``,
                positive integer ``scale_max``, and ``width``.

        The three mappings are not modified.
        """
        super().__init__(point, codec_config, mul_config, add_config)
        #: Largest runtime carry scale accepted by every FFT stage.
        self.scale_max = self.core.scale_max


    def _make_core(self, point, mul_config, add_config):
        """Return the dynamic-scale streaming transform this class wraps."""
        return fft_dyn(point, dict(mul_config), dict(add_config))


    def _reset(self):
        """Reset state owned directly by this class.

        This class holds no local mutable state beyond the core's reported
        compensation, which the inherited reset clears. This hook returns
        ``None``.
        """
        pass


    def forward(self, input_real, input_imag, scales):
        """Process one timestep at the requested runtime stage scales.

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
            input shape and this call's core compensation applied.

        The call advances both encoders, the core, both decoders, and this
        class's ``timestep_cur`` once. Invalid shapes or scales raise before any
        child advances. Hold the scale list constant after reset so that one
        compensation factor describes the whole decoded run.
        """
        self._check_shapes(input_real, input_imag)
        self.core._runtime_scales(scales)
        output_real_spike, output_imag_spike = self.core(
            self.encode_real(input_real), self.encode_imag(input_imag), scales
        )
        self.decode_real(output_real_spike)
        self.decode_imag(output_imag_spike)
        compensation = self.core.compensation
        return (self.decode_real.spike_value * compensation,
                self.decode_imag.spike_value * compensation)

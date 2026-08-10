import math

import torch
from loguru import logger

from .butterfly_ugemm_dyn import butterfly_ugemm_dyn
from .fft import fft


class fft_dyn(fft):
    r"""Evaluate a fully streaming radix-2 FFT with one runtime scale per stage.

    Use this class when the per-stage carry scales are chosen while the stream
    runs and the transform must stay in the spike domain: it takes bipolar 0/1
    spike tensors and returns bipolar 0/1 spike tensors and holds no encoder and
    no decoder, so it is not the numeric-port class :class:`fft_dyn_hub`.

    A scale list held constant from reset approximates the analytic FFT when at
    most one stage uses scale 1. Lists with two or more scale-1 stages remain
    legal but can clip because a scale-1 adder drains at most one unit per
    timestep against a fan-in of three. The scale list passed to a call holds the
    requested per-stage scales, and the first stage's internal adder runs at
    twice the requested ``scales[0]``, so that value must not exceed half of
    ``scale_max``. This class exposes no ``scales`` attribute, because its
    scales arrive as call inputs.
    After a scale change without reset, the output stream mixes segments
    produced under different compensations and :attr:`compensation` describes
    only the most recent call.

    As in :class:`fft`, all stages share one bit-identical weight sequence and
    decorrelate through the conditional sequence-index advance on distinct data.
    """


    def __init__(self, point, mul_config, add_config):
        """Configure the FFT size and maximum runtime adder scale.

        Args:
            point: Positive power-of-two transform length.
            mul_config: Bipolar conditional-spike multiplier configuration
                containing ``polarity``, ``timestep``, and ``generator``.
            add_config: Dynamic adder configuration containing ``polarity``,
                positive integer ``scale_max``, and ``width``. The width must
                satisfy the bound documented on :class:`butterfly_ugemm` for
                twice ``scale_max``, the largest scale the first stage can be
                built for.
        """
        if 'scale_max' not in add_config:
            message = 'Missing key <scale_max> in the dynamic adder configuration.'
            logger.error(message)
            raise AssertionError(message)
        scale_max = add_config['scale_max']
        if type(scale_max) is not int or scale_max < 1:
            message = f'fft_dyn scale_max must be a positive int: got <{scale_max}>.'
            logger.error(message)
            raise AssertionError(message)
        static_add_config = dict(add_config)
        static_add_config['scale'] = static_add_config.pop('scale_max')
        super().__init__(point, mul_config, static_add_config)

        for stage in range(self.stages):
            static_stage = getattr(self, f'butterfly_stage_{stage}')
            lane = static_stage.lane
            setattr(
                self,
                f'butterfly_stage_{stage}',
                butterfly_ugemm_dyn(
                    static_stage.twiddle_stack.narrow(0, 0, lane),
                    static_stage.twiddle_stack.narrow(0, 2 * lane, lane),
                    dict(mul_config),
                    dict(add_config),
                ),
            )
        # fft_dyn removes fft.scales because its stage scales are runtime inputs.
        del self.scales
        #: Largest runtime carry scale accepted by every FFT stage.
        self.scale_max = self.butterfly_stage_0.scale_max
        #: Factor the output streams of the most recent call are divided by.
        self.compensation = None


    def _reset(self):
        """Clear the reported compensation after inherited dynamic stages reset.

        The reported factor returns to ``None`` because no call has selected a
        scale list yet. This hook returns ``None``.
        """
        super()._reset()
        self.compensation = None


    def forward(self, input_real_spike, input_imag_spike, scales):
        """Process one FFT timestep using the requested runtime stage scales.

        Args:
            input_real_spike: Real-part spike tensor shaped ``(point, ...)`` in
                natural sample order.
            input_imag_spike: Imaginary-part spike tensor with the same shape as
                ``input_real_spike``.
            scales: One Python int broadcast to all stages, or a Python list
                containing exactly one int per stage. Every value must be in
                ``1`` through ``scale_max``, twice the first value must not
                exceed ``scale_max``, and bool is rejected.

        Returns:
            Natural-order ``(output_real_spike, output_imag_spike)`` as bipolar
            0/1 spike tensors with the input shape, each encoding the spectrum
            divided by :attr:`compensation`.

        Held scales support analytic FFT fidelity when at most one stage uses
        scale 1. After a change, the stream mixes segments accumulated under the
        prior compensation, so decode a run whose scales never changed.
        """
        runtime_scales = self._runtime_scales(scales)
        if input_real_spike.shape != input_imag_spike.shape:
            message = (
                f'FFT input shapes must match: got <{input_real_spike.shape}> and '
                f'<{input_imag_spike.shape}>.'
            )
            logger.error(message)
            raise AssertionError(message)
        if input_real_spike.ndim == 0 or input_real_spike.shape[0] != self.point:
            message = (
                f'FFT first input dimension must equal point <{self.point}>: '
                f'got shape <{input_real_spike.shape}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        current_real = input_real_spike.index_select(0, self._bit_reversed)
        current_imag = input_imag_spike.index_select(0, self._bit_reversed)
        for stage, scale in enumerate(runtime_scales):
            first = getattr(self, f'_first_indices_{stage}')
            second = getattr(self, f'_second_indices_{stage}')
            butterfly = getattr(self, f'butterfly_stage_{stage}')
            # Halving the samples ahead of stage 0 would need a numeric port the
            # spike interface does not have, so the same attenuation is folded
            # into that stage's carry scale, which divides exactly.
            y0r, y0i, y1r, y1i = butterfly(
                current_real.index_select(0, first),
                current_imag.index_select(0, first),
                current_real.index_select(0, second),
                current_imag.index_select(0, second),
                scale * 2 if stage == 0 else scale,
            )
            next_real = torch.empty_like(current_real)
            next_imag = torch.empty_like(current_imag)
            next_real.index_copy_(0, first, y0r)
            next_real.index_copy_(0, second, y1r)
            next_imag.index_copy_(0, first, y0i)
            next_imag.index_copy_(0, second, y1i)
            current_real, current_imag = next_real, next_imag

        self.compensation = 2 * math.prod(runtime_scales)
        return current_real, current_imag


    def __call__(self, input_real_spike, input_imag_spike, scales):
        """Validate scale controls, then process one FFT timestep.

        Args:
            input_real_spike: Real-part spike tensor shaped ``(point, ...)``.
            input_imag_spike: Imaginary-part spike tensor with the same shape.
            scales: Exact Python int broadcast to all stages, or an exact-length
                list of Python ints in ``1`` through ``scale_max``.

        Returns:
            Natural-order ``(output_real_spike, output_imag_spike)``.

        Invalid controls raise before this FFT, its stages, or their adders
        advance or update state. A valid call advances each exactly once.
        """
        self._runtime_scales(scales)
        return super().__call__(input_real_spike, input_imag_spike, scales)


    def _runtime_scales(self, scales):
        """Return a validated copy of the runtime per-stage scale list."""
        if type(scales) is int:
            runtime_scales = [scales] * self.stages
        elif isinstance(scales, list):
            if len(scales) != self.stages:
                message = (
                    f'FFT runtime scale list length <{len(scales)}> must equal '
                    f'the stage count <{self.stages}>.'
                )
                logger.error(message)
                raise AssertionError(message)
            runtime_scales = list(scales)
        else:
            message = f'FFT runtime scales must be a Python int or list: got <{scales}>.'
            logger.error(message)
            raise AssertionError(message)
        for stage, scale in enumerate(runtime_scales):
            if type(scale) is not int:
                message = f'FFT runtime scale at stage <{stage}> must be an int: got <{scale}>.'
                logger.error(message)
                raise AssertionError(message)
            if scale < 1 or scale > self.scale_max:
                message = (
                    f'FFT runtime scale <{scale}> at stage <{stage}> outside '
                    f'the supported range <1> to scale_max <{self.scale_max}>.'
                )
                logger.error(message)
                raise AssertionError(message)
        if 2 * runtime_scales[0] > self.scale_max:
            message = (
                f'FFT runtime scale <{runtime_scales[0]}> at stage <0> must not exceed '
                f'half of scale_max <{self.scale_max}>, since stage 0 runs at twice '
                f'its requested scale.'
            )
            logger.error(message)
            raise AssertionError(message)
        return runtime_scales

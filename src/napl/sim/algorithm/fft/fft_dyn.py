import math

import torch
from loguru import logger

from napl.sim.base import napl_base

from .butterfly_mix import butterfly_mix
from .butterfly_mix_dyn import butterfly_mix_dyn
from .butterfly_ugemm import butterfly_ugemm
from .butterfly_ugemm_dyn import butterfly_ugemm_dyn
from .fft import TWIDDLE_DIM_FIRST


class fft_dyn(napl_base):
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

    The optional ``mul_config`` key ``kernel`` selects the stage class:
    ``'ugemm'``, the default, builds :class:`butterfly_ugemm_dyn` stages, and
    ``'mix'`` builds :class:`butterfly_mix_dyn` stages. The key is consumed here
    and never reaches a stage.

    With ``'ugemm'``, as in :class:`fft`, all stages share one bit-identical
    weight sequence and decorrelate through the conditional sequence-index
    advance on distinct data. With ``'mix'``, each stage encodes its own twiddle
    stream, and stage ``index`` takes the one-based Sobol dimension
    ``5 + index``. That schedule gives every stage a distinct twiddle dimension
    and clears the dimensions ``1`` through ``4`` conventionally used by the
    input encoders, so no stage's twiddle stream correlates with its own input
    stream or with another stage's twiddle stream. Because that schedule
    decorrelates by Sobol dimension, ``'mix'`` requires a sobol-family generator
    (``'sobol'``, ``'rc'``, or ``'rate'``) in ``mul_config``; the stage rejects
    any other generator.
    """


    def __init__(self, point, mul_config, add_config):
        """Configure the FFT size and maximum runtime adder scale.

        Args:
            point: Positive power-of-two transform length.
            mul_config: Bipolar multiplier configuration containing
                ``polarity``, ``timestep``, and ``generator``, plus the optional
                ``kernel``, which is ``'ugemm'`` or ``'mix'`` and defaults to
                ``'ugemm'``.
            add_config: Dynamic adder configuration containing ``polarity``,
                positive ``scale_max``, ``intwidth``, and ``fracwidth``. The
                accumulator must satisfy the bound documented on
                :class:`butterfly_ugemm` for twice ``scale_max``, the largest
                scale the first stage can be built for.
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
        super().__init__(mul_config, ['polarity', 'timestep', 'generator'],
                         optional_key_list=['kernel'], polarity_required=True)
        kernel = mul_config.get('kernel', 'ugemm')
        kernel = kernel.lower() if isinstance(kernel, str) else kernel
        if kernel not in ('ugemm', 'mix'):
            message = (
                f'Invalid kernel: <{kernel!r}>; legal values: <[\'mix\', \'ugemm\']>.'
            )
            logger.error(message)
            raise AssertionError(message)
        stage_mul_config = dict(mul_config)
        stage_mul_config.pop('kernel', None)
        if isinstance(point, bool) or not isinstance(point, int) \
                or point < 2 or point & (point - 1):
            message = (
                f'Invalid point: <{point!r}>; legal values: a power-of-two integer '
                f'greater than or equal to 2.'
            )
            logger.error(message)
            raise AssertionError(message)
        if self.polarity != 'bipolar':
            message = f'Invalid polarity: <{self.polarity}>; legal values: <[\'bipolar\']>.'
            logger.error(message)
            raise AssertionError(message)

        #: Number of samples in each transform.
        self.point = point
        #: Number of radix-2 stages in the transform.
        self.stages = int(math.log2(point))

        bit_width = self.stages
        bit_reversed = [
            int(f'{index:0{bit_width}b}'[::-1], 2)
            for index in range(point)
        ]
        self.register_buffer('_bit_reversed', torch.tensor(bit_reversed, dtype=torch.long))

        static_stages = []
        stage_mul_configs = []
        for stage in range(self.stages):
            span = 2 ** (stage + 1)
            half = span // 2
            first = []
            second = []
            twiddle = []
            for block in range(0, point, span):
                for offset in range(half):
                    first.append(block + offset)
                    second.append(block + offset + half)
                    angle = -2 * math.pi * offset / span
                    twiddle.append(complex(math.cos(angle), math.sin(angle)))
            twiddle_tensor = torch.tensor(twiddle, dtype=torch.complex64)
            self.register_buffer(
                f'_first_indices_{stage}', torch.tensor(first, dtype=torch.long)
            )
            self.register_buffer(
                f'_second_indices_{stage}', torch.tensor(second, dtype=torch.long)
            )
            stage_add_config = dict(static_add_config)
            # Halving the samples ahead of stage 0 would need a numeric port the
            # spike interface does not have, so the same attenuation is folded
            # into that stage's carry scale, which divides exactly.
            stage_add_config['scale'] = scale_max * (2 if stage == 0 else 1)
            if kernel == 'mix':
                # Each Gaines stage encodes its own twiddle, so it gets its own
                # Sobol dimension above the input encoders' dimensions 1 to 4.
                stage_mul_configs.append(dict(stage_mul_config, dim=TWIDDLE_DIM_FIRST + stage))
                static_class = butterfly_mix
            else:
                stage_mul_configs.append(dict(stage_mul_config))
                static_class = butterfly_ugemm
            try:
                # The static stage is a construction-time check that the largest
                # runtime scale fits the requested accumulator width.
                static_stages.append(static_class(
                    twiddle_tensor.real,
                    twiddle_tensor.imag,
                    dict(stage_mul_configs[stage]),
                    stage_add_config,
                ))
            except AssertionError as error:
                message = f'FFT stage <{stage}> construction failed: {error}'
                logger.error(message)
                raise AssertionError(message) from error

        dyn_class = butterfly_mix_dyn if kernel == 'mix' else butterfly_ugemm_dyn
        for stage, static_stage in enumerate(static_stages):
            lane = static_stage.lane
            setattr(
                self,
                f'butterfly_stage_{stage}',
                dyn_class(
                    static_stage.twiddle_stack.narrow(0, 0, lane),
                    static_stage.twiddle_stack.narrow(0, 2 * lane, lane),
                    dict(stage_mul_configs[stage]),
                    dict(add_config),
                ),
            )
        #: Largest runtime carry scale accepted by every FFT stage.
        self.scale_max = self.butterfly_stage_0.scale_max
        #: Factor the output streams of the most recent call are divided by.
        self.compensation = None

        #: Hardware latency and timing metadata for the streaming FFT.
        self.hw.pp_delay = 0
        #: Whether the RTL counterpart must hold its own encoder, true when any part does.
        self.internal_encode = any(part.internal_encode for part in self.children())
        #: Rate coding on both input and both output spike ports.
        self.encoding_io = {port: 'rc' for port in
                            ('input_real', 'input_imag',
                             'output_real', 'output_imag')}
        #: Stream polarity of the four spike ports, which share the class polarity.
        self.polarity_io = {port: self.polarity for port in
                            ('input_real', 'input_imag',
                             'output_real', 'output_imag')}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Clear the reported compensation after the dynamic stages reset.

        The reported factor returns to ``None`` because no call has selected a
        scale list yet. This hook returns ``None``.
        """
        self.compensation = None


    def forward(self, input_real, input_imag, scales):
        """Process one FFT timestep using the requested runtime stage scales.

        Args:
            input_real: Real-part spike tensor shaped ``(point, ...)`` in
                natural sample order.
            input_imag: Imaginary-part spike tensor with the same shape as
                ``input_real``.
            scales: One Python int broadcast to all stages, or a Python list
                containing exactly one int per stage. Every value must be in
                ``1`` through ``scale_max``, twice the first value must not
                exceed ``scale_max``, and bool is rejected.

        Returns:
            Natural-order ``(output_real, output_imag)`` as bipolar
            0/1 spike tensors with the input shape, each encoding the spectrum
            divided by :attr:`compensation`.

        Held scales support analytic FFT fidelity when at most one stage uses
        scale 1. After a change, the stream mixes segments accumulated under the
        prior compensation, so decode a run whose scales never changed.
        """
        runtime_scales = self._runtime_scales(scales)
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

        current_real = input_real.index_select(0, self._bit_reversed)
        current_imag = input_imag.index_select(0, self._bit_reversed)
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


    def __call__(self, input_real, input_imag, scales):
        """Validate scale controls, then process one FFT timestep.

        Args:
            input_real: Real-part spike tensor shaped ``(point, ...)``.
            input_imag: Imaginary-part spike tensor with the same shape.
            scales: Exact Python int broadcast to all stages, or an exact-length
                list of Python ints in ``1`` through ``scale_max``.

        Returns:
            Natural-order ``(output_real, output_imag)``.

        Invalid controls raise before this FFT, its stages, or their adders
        advance or update state. A valid call advances each exactly once.
        """
        self._runtime_scales(scales)
        return super().__call__(input_real, input_imag, scales)


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

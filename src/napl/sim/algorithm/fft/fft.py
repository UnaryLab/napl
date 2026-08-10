import math

import torch
from loguru import logger

from napl.sim.base import napl_base

from .butterfly_ugemm import butterfly_ugemm


class fft(napl_base):
    r"""Evaluate a radix-2 decimation-in-time FFT with spike streams end to end.

    Use this class for a power-of-two complex FFT that never leaves the spike
    domain: it takes bipolar 0/1 spike tensors and returns bipolar 0/1 spike
    tensors, one timestep per call, and contains no encoder and no decoder. A
    caller encodes the samples upstream and decodes the spectrum downstream.
    This is the difference from :class:`fft_hub`, which wraps this class and
    carries numeric values on its ports because it encodes its inputs and
    progressively decodes its outputs at the transform boundary.

    The input spike streams are supplied in natural order and are reordered
    internally. The first dimension is the transform dimension and must equal
    ``point``; any trailing dimensions are independent transforms processed in
    parallel.

    A stream cannot represent a magnitude above one, while the spectrum of a
    full-scale input grows with ``point``, so the returned streams encode the
    spectrum divided by :attr:`compensation`. The class does not undo that
    division; a consumer that wants the spectrum multiplies the decoded stream
    value by :attr:`compensation`.

    The positive integer adder scale may be shared by all stages or supplied as
    one value per stage. The first stage's adder runs at twice its requested
    scale, which supplies the headroom the growing spectrum needs, so its width
    must satisfy the bound documented on :class:`butterfly_ugemm` for that
    doubled scale. :attr:`scales` reports the requested per-stage scales, so
    stage 0's internal adder runs at twice ``scales[0]`` while every later stage
    runs at its reported value.

    All stages share one weight sequence: their multipliers accept no sequence
    dimension and are built from the same ``mul_config``, so their number
    sequences are bit-identical. Decorrelation across stages comes from
    ``mul_ugemm`` advancing its sequence index only on enabling timesteps, which
    makes the stages' index pointers diverge once they see distinct data.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import decode, encode
        from napl.sim.algorithm.fft import fft

        codec = {'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol'}
        adder = {'polarity': 'bipolar', 'scale': 2, 'width': 8}
        operation = fft(8, codec, adder)
        source_real, source_imag = encode(codec), encode(dict(codec, dim=2))
        sink = decode(codec)
        real = torch.zeros(8, 1)
        imag = torch.zeros_like(real)
        for _ in range(256):
            spectrum_real, spectrum_imag = operation(source_real(real),
                                                     source_imag(imag))
            sink(spectrum_real)
        print(sink.spike_value * operation.compensation)

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """


    def __init__(self, point, mul_config, add_config):
        """Configure the transform size and the components used by every stage.

        Args:
            point: Positive power-of-two transform length.
            mul_config: Bipolar conditional-spike multiplier configuration
                containing ``polarity``, ``timestep``, and ``generator``.
            add_config: Bipolar scaled-adder configuration. Its ``scale`` may be
                one positive Python int shared by every stage, or a list
                containing one positive Python int per stage.

        The two mappings accept the same keys as
        :class:`~napl.sim.algorithm.fft.butterfly_ugemm` and are not modified.
        """
        super().__init__(mul_config, ['polarity', 'timestep', 'generator'],
                         polarity_required=True)
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

        if 'scale' not in add_config:
            message = 'Missing key <scale> in the input adder configuration.'
            logger.error(message)
            raise AssertionError(message)
        requested_scale = add_config['scale']
        if isinstance(requested_scale, list):
            if len(requested_scale) != self.stages:
                message = (
                    f'FFT scale list length <{len(requested_scale)}> must equal '
                    f'the stage count <{self.stages}>.'
                )
                logger.error(message)
                raise AssertionError(message)
            requested_scales = list(requested_scale)
        else:
            requested_scales = [requested_scale] * self.stages
        for stage, scale in enumerate(requested_scales):
            if type(scale) is not int or scale < 1:
                message = (
                    f'Invalid FFT scale <{scale!r}> at stage <{stage}>; legal '
                    f'values are positive Python ints.'
                )
                logger.error(message)
                raise AssertionError(message)

        #: Requested carry scale per stage; stage 0's adder runs at twice scales[0].
        self.scales = requested_scales
        #: Factor the output streams are divided by, ``2 * prod(scales)``.
        self.compensation = 2 * math.prod(self.scales)

        bit_width = self.stages
        bit_reversed = [
            int(f'{index:0{bit_width}b}'[::-1], 2)
            for index in range(point)
        ]
        self.register_buffer('_bit_reversed', torch.tensor(bit_reversed, dtype=torch.long))

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
            stage_add_config = dict(add_config)
            # Halving the samples ahead of stage 0 would need a numeric port the
            # spike interface does not have, so the same attenuation is folded
            # into that stage's carry scale, which divides exactly.
            stage_add_config['scale'] = requested_scales[stage] * (2 if stage == 0 else 1)
            try:
                stage_butterfly = butterfly_ugemm(
                    twiddle_tensor.real,
                    twiddle_tensor.imag,
                    dict(mul_config),
                    stage_add_config,
                )
            except AssertionError as error:
                message = f'FFT stage <{stage}> construction failed: {error}'
                logger.error(message)
                raise AssertionError(message) from error
            setattr(self, f'butterfly_stage_{stage}', stage_butterfly)

        #: Hardware latency and timing metadata for the streaming FFT.
        self.hw.pp_delay = 0
        #: Rate coding on both input and both output spike ports.
        self.encoding_io = {port: 'rc' for port in
                            ('input_real_spike', 'input_imag_spike',
                             'output_real_spike', 'output_imag_spike')}
        #: Stream polarity of the four spike ports, which share the class polarity.
        self.polarity_io = {port: self.polarity for port in
                            ('input_real_spike', 'input_imag_spike',
                             'output_real_spike', 'output_imag_spike')}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset class-local execution state.

        The FFT holds no local mutable state. The inherited reset method resets
        the timestep and every stage butterfly before this hook returns ``None``.
        """
        pass


    def forward(self, input_real_spike, input_imag_spike):
        """Process one spike timestep of a batch of complex transforms.

        Args:
            input_real_spike: Real-part spike tensor shaped ``(point, ...)`` in
                natural sample order.
            input_imag_spike: Imaginary-part spike tensor with the same shape as
                ``input_real_spike``.

        Returns:
            ``(output_real_spike, output_imag_spike)`` as bipolar 0/1 spike
            tensors in natural FFT-bin order with the input shape, each encoding
            the spectrum divided by :attr:`compensation`.

        The call advances every stage once and does not modify either input.
        """
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

        for stage in range(self.stages):
            first = getattr(self, f'_first_indices_{stage}')
            second = getattr(self, f'_second_indices_{stage}')
            butterfly = getattr(self, f'butterfly_stage_{stage}')
            y0r, y0i, y1r, y1i = butterfly(
                current_real.index_select(0, first),
                current_imag.index_select(0, first),
                current_real.index_select(0, second),
                current_imag.index_select(0, second),
            )
            next_real = torch.empty_like(current_real)
            next_imag = torch.empty_like(current_imag)
            next_real.index_copy_(0, first, y0r)
            next_real.index_copy_(0, second, y1r)
            next_imag.index_copy_(0, first, y0i)
            next_imag.index_copy_(0, second, y1i)
            current_real, current_imag = next_real, next_imag

        return current_real, current_imag

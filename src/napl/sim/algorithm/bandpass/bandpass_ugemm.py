import math

import torch
from loguru import logger

from napl.sim.base import napl_base
from napl.sim.operation import add_scale, delay, mul_ugemm


class bandpass_ugemm(napl_base):
    r"""Apply a fixed finite-impulse-response bandpass filter to a spike stream.

    Use this class with bipolar rate-coded input and fixed real taps designed
    before construction. One call consumes one input-spike timestep and returns
    one output-spike timestep. The returned stream represents the FIR result
    divided by :attr:`compensation`.

    Every tap multiplier uses the same bit-identical number sequence because
    ``mul_ugemm`` exposes no per-instance sequence dimension. Its conditional
    indices can diverge when the delayed tap streams differ, but the tap
    products are not explicitly decorrelated.

    Delay registers fill with zero bits. A zero bit represents ``-1`` in
    bipolar rate coding, so the first ``tap_count - 1`` outputs include a
    non-neutral pre-history warmup transient.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.algorithm.bandpass import bandpass_ugemm

        taps = torch.tensor([0.25, 0.0, -0.5, 0.0, 0.25])
        config = {'polarity': 'bipolar', 'timestep': 256,
                  'generator': 'sobol', 'scale': 1.0,
                  'intwidth': 10, 'fracwidth': 8}
        operation = bandpass_ugemm(taps, config)
        output = operation(torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *Discrete-Time Signal Processing*, 3rd ed., 2010.
    """
    streaming = True


    def __init__(self, taps, config):
        """Configure fixed taps, conditional multipliers, delays, and accumulation.

        Args:
            taps: Non-empty one-dimensional tensor of finite bipolar
                coefficients in ``[-1, 1]``.
            config: Configuration containing ``polarity``, ``timestep``,
                ``generator``, ``scale``, ``intwidth``, and ``fracwidth``.
                ``scale`` is quantized on the ``2 ** -fracwidth`` grid and the
                effective output compensation must be at least the L1 norm of
                ``taps`` so the normalized FIR result stays in the bipolar range.
        """
        super().__init__(
            config,
            ['polarity', 'timestep', 'generator', 'scale', 'intwidth', 'fracwidth'],
            polarity_required=True,
        )
        if self.polarity != 'bipolar':
            message = f"Invalid polarity: <{self.polarity}>; legal values: <['bipolar']>."
            logger.error(message)
            raise AssertionError(message)
        if not isinstance(taps, torch.Tensor) or taps.ndim != 1 or taps.numel() == 0:
            description = type(taps).__name__ if not isinstance(taps, torch.Tensor) else tuple(taps.shape)
            message = (
                f'Invalid taps: <{description}>; legal values: a non-empty 1-D tensor.'
            )
            logger.error(message)
            raise AssertionError(message)
        if not torch.isfinite(taps).all() or torch.any(taps.abs() > 1):
            message = 'Invalid taps: all coefficients must be finite and in [-1, 1].'
            logger.error(message)
            raise AssertionError(message)
        requested_scale = config['scale']
        tap_norm = taps.detach().abs().sum().item()
        if type(requested_scale) not in (int, float) or not math.isfinite(requested_scale) \
                or requested_scale <= 0 or requested_scale < tap_norm:
            message = (
                f'Invalid scale: <{requested_scale}>; legal values: a positive finite '
                f'number at least <{tap_norm}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Fixed FIR coefficients in lag order, starting with the current input.
        self.taps: torch.Tensor
        self.register_buffer('taps', taps.detach().clone().type(self.ntype))
        #: Number of FIR coefficients.
        self.tap_count = taps.numel()
        mul_config = {
            'polarity': self.polarity,
            'timestep': config['timestep'],
            'generator': config['generator'],
        }
        for index in range(1, self.tap_count):
            setattr(self, f'delay_{index}', delay({'depth': 1, 'polarity': self.polarity}))
        for index in range(self.tap_count):
            setattr(self, f'mul_{index}', mul_ugemm(mul_config))
        self.acc = add_scale({
            'polarity': self.polarity,
            'scale': requested_scale,
            'intwidth': config['intwidth'],
            'fracwidth': config['fracwidth'],
        })
        if self.acc.scale < tap_norm:
            message = (
                f'Quantized scale <{self.acc.scale}> is below the tap L1 norm '
                f'<{tap_norm}>; increase scale or fracwidth.'
            )
            logger.error(message)
            raise AssertionError(message)
        #: Effective scale by which the output stream divides the FIR result.
        self.compensation = self.acc.scale

        #: Hardware latency and timing metadata for the streaming composition.
        self.hw.pp_delay = 0
        #: Encoder the hardware counterpart carries, private when any part carries one.
        self.internal_encode = 'private' if any(part.internal_encode != 'none' for part in self.children()) else 'none'
        #: Rate coding on the input and output spike ports.
        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        #: Bipolar polarity on the input and output spike ports.
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """Reset no class-local mutable state.

        The inherited reset method resets the timestep, delays, multipliers,
        and accumulator before this hook returns ``None``.
        """
        pass


    def forward(self, input):
        """Process one input-spike timestep.

        Args:
            input: Current bipolar 0/1 spike tensor of any shape.

        Returns:
            One bipolar output-spike tensor with the input shape. The call
            advances every delay, multiplier, and the output accumulator once.
        """
        tap_inputs = [input]
        delayed = input
        for index in range(1, self.tap_count):
            delayed = getattr(self, f'delay_{index}')(delayed)
            tap_inputs.append(delayed)

        products = [
            getattr(self, f'mul_{index}')(tap_input, self.taps[index])
            for index, tap_input in enumerate(tap_inputs)
        ]
        return self.acc(torch.stack(products), dim=0)

import torch

from loguru import logger
from napl.sim.base import napl_base


class clamp(napl_base):
    r"""
    Clamp a rate-coded stream to a fixed ``[lo, hi]`` band with a saturating counter.

    Use this streaming operation to bound a single rate-coded input to a
    configurable band. The target rate-domain operation is

    .. math::

       y = \min(\max(x, \mathit{lo}), \mathit{hi}).

    This is a *derived* operation: the band generalization of :class:`relu_sat`,
    whose saturating counter clamps the running estimate at the single floor
    ``0`` to realize :math:`\max(x, 0)`. Here the counter clamps the running
    estimate to the two-sided band ``[lo, hi]``, both fixed-point constants on a
    ``2 ** -fracwidth`` grid and both inside the polarity's legal value range.
    Unipolar and bipolar rate-coded inputs are supported; the input and the
    output share the configured polarity. One input stream is consumed and one
    output stream is produced.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import clamp

        operation = clamp({'polarity': 'bipolar', 'lo': -0.5, 'hi': 0.5})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """


    def __init__(
            self,
            config={
                'polarity': 'bipolar',
                'lo': -0.5,
                'hi': 0.5,
            }
        ):
        """
        Configure the polarity, the fixed-point band, and its grid.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, either ``"unipolar"`` or ``"bipolar"``.
              - **lo**: Lower band bound, a number inside the polarity's legal value range and below **hi**, quantized to the nearest multiple of ``2 ** -fracwidth``.
              - **hi**: Upper band bound, a number inside the polarity's legal value range and above **lo**, quantized to the nearest multiple of ``2 ** -fracwidth``.
              - **fracwidth**: Fractional bits of the band grid; the default is ``8``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'lo', 'hi'], optional_key_list=['fracwidth'], polarity_required=True)

        #: Fractional bits of the fixed-point band grid.
        self.fracwidth = config.get('fracwidth', 8)
        if type(self.fracwidth) is not int or self.fracwidth < 0:
            message = f'Invalid fracwidth: <{self.fracwidth}>; legal values: a non-negative integer.'
            logger.error(message)
            raise AssertionError(message)
        #: Smallest value the band grid resolves, ``2 ** -fracwidth``.
        self.grid = 2.0 ** (-self.fracwidth)

        # Bipolar rate coding spans values [-1, 1]; unipolar spans [0, 1].
        legal_low = -1.0 if self.polarity == 'bipolar' else 0.0
        legal_high = 1.0
        lo = config['lo']
        hi = config['hi']
        for label, bound in (('lo', lo), ('hi', hi)):
            if type(bound) not in (int, float) or bound < legal_low or bound > legal_high:
                message = (f'Invalid {label}: <{bound}>; legal values: a number in '
                           f'[{legal_low}, {legal_high}] for {self.polarity} polarity.')
                logger.error(message)
                raise AssertionError(message)
        #: Lower band bound quantized to the grid.
        self.lo = round(float(lo) / self.grid) * self.grid
        #: Upper band bound quantized to the grid.
        self.hi = round(float(hi) / self.grid) * self.grid
        if self.lo >= self.hi:
            message = f'Invalid band: lo <{self.lo}> must be strictly below hi <{self.hi}>.'
            logger.error(message)
            raise AssertionError(message)

        # Value contribution of one input spike and the emit rate of a clamped
        # value, both as affine maps of the polarity's spike/value convention.
        if self.polarity == 'bipolar':
            self.val_scale, self.val_offset = 2.0, -1.0
            self.rate_scale, self.rate_offset = 0.5, 0.5
        else:
            self.val_scale, self.val_offset = 1.0, 0.0
            self.rate_scale, self.rate_offset = 1.0, 0.0

        #: Running sum of per-timestep input value contributions.
        self.accumulator: torch.Tensor
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
        #: Cumulative count of emitted output spikes.
        self.emitted: torch.Tensor
        self.register_buffer('emitted', torch.zeros(1, dtype=self.ntype))
        #: Hardware latency and timing metadata for the saturating-counter path.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Clear the running-estimate accumulator and the emitted-spike counter,
        arming both buffers to broadcast to the next input shape.
        """
        self.accumulator.resize_(1).zero_()
        self.emitted.resize_(1).zero_()


    def forward(self, input):
        """
        Process one timestep of a rate-coded input stream.

        Args:
            input: Tensor of current 0/1 input spikes.

        Returns:
            Output spike tensor with the same shape as ``input``, whose stream
            decodes to ``clamp(input, lo, hi)``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        # Accumulate the input value contribution, so accumulator / timestep is
        # the running estimate of the input's rate-domain value.
        value_step = input.type(self.ntype).mul(self.val_scale).add(self.val_offset)
        if self.accumulator.shape == value_step.shape:
            self.accumulator.add_(value_step)
        else:
            updated = self.accumulator.add(value_step)
            self.accumulator.resize_as_(updated).copy_(updated.detach())

        # The saturating counter clamps the running estimate to the [lo, hi] band.
        estimate = self.accumulator.div(self.timestep_cur).clamp_(self.lo, self.hi)
        # Emit output spikes to track the clamped value's cumulative target count.
        target_count = estimate.mul(self.rate_scale).add(self.rate_offset).mul_(self.timestep_cur)
        output = torch.ge(target_count.sub_(self.emitted), 0.5).type(self.ntype)
        if self.emitted.shape == output.shape:
            self.emitted.add_(output)
        else:
            updated = self.emitted.add(output)
            self.emitted.resize_as_(updated).copy_(updated.detach())
        return output.type(self.stype)

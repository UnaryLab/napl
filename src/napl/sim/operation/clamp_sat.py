import math
import torch

from loguru import logger
from napl.sim.base import napl_base
from .add_scale import add_scale
from .encode import gen_num_seq


class clamp_sat(napl_base):
    r"""
    Clamp a rate-coded stream to a fixed ``[lo, hi]`` band with saturating adders.

    Use this streaming operation to bound a single rate-coded input to a
    configurable band. The target rate-domain operation is

    .. math::

       y = \min(\max(x, \mathit{lo}), \mathit{hi}).

    Three chained bipolar saturating adders realize the band. The legal band is
    the whole value range of the polarity: ``lo`` and ``hi`` both inside the
    range, with ``lo`` strictly below ``hi``. Both bounds are fixed-point
    constants placed on a ``2 ** -fracwidth`` grid.

    The band is approximate: the output rate departs from the requested band,
    with the error reported by the printed output of the focused test
    ``tests/operation/test_clamp_sat.py``. Placing the bounds on the grid also
    leaves an asymmetric band a DC offset of up to half a grid step against the
    requested bounds, which does not shrink as the timestep count grows. The
    stage accumulators are fixed at 4 integer bits, sized for a low-discrepancy
    (Sobol-class) input stream; a bursty or highly correlated input stream
    exceeds that width, so a caller feeding one should expect the band to be
    placed less accurately.

    Unipolar and bipolar rate-coded inputs are supported; the input and the
    output share the configured polarity. A unipolar band is clamped by the same
    bipolar circuit on the same spikes, which read as the bipolar value
    :math:`2p - 1`, against the mapped band ``(2 * lo - 1, 2 * hi - 1)``. One
    input stream is consumed and one output stream is produced.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import clamp_sat

        operation = clamp_sat({'polarity': 'bipolar', 'lo': -0.5, 'hi': 0.5})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """
    #: Dominant hardware mechanism of this class.
    mechanism = 'integrate-and-fire'


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
              - **dim**: One-based Sobol dimension of the first band-constant stream, with the other two on ``dim + 1`` and ``dim + 2``; the default is ``2``, and the largest legal value is ``21199``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'lo', 'hi'], optional_key_list=['fracwidth', 'dim'], polarity_required=True)

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
            # The comparisons below are both False for a NaN bound and round() raises on
            # a non-finite value, so the finiteness test runs before either.
            if (type(bound) not in (int, float) or not math.isfinite(bound)
                    or bound < legal_low or bound > legal_high):
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

        # The stages are bipolar, so a unipolar band moves to the bipolar value it
        # denotes on the same spikes: a unipolar rate p reads as the bipolar value
        # 2p - 1.
        if self.polarity == 'bipolar':
            band_lo, band_hi = self.lo, self.hi
        else:
            band_lo, band_hi = 2.0 * self.lo - 1.0, 2.0 * self.hi - 1.0

        # Each stage is a three-entry add_scale of unit scale whose rate saturation
        # clips its bipolar output value to [-1, 1], so on an input of value v the
        # stages compute
        #     u = sat(v - (1 + lo)),
        #     w = sat(u + (2 + lo - hi)),
        #     y = sat(w + (hi - 1)).
        # The first stage's low saturation lands at the floor, since v - (1 + lo) <= -1
        # holds exactly when v <= lo, so u = max(v, lo) - (1 + lo). The second stage's
        # high saturation lands at the ceiling, since u + (2 + lo - hi) = max(v, lo) +
        # 1 - hi >= 1 holds exactly when max(v, lo) >= hi, so w = min(max(v, lo), hi) +
        # 1 - hi. The second stage never saturates low and the third never saturates at
        # all, because hi <= lo + 2 and the banded value already lies inside [-1, 1].
        # The third stage shifts the band back, so y is the clamped value.
        # Each stage carries one input entry, one constant stream, and one fixed entry,
        # the fixed entry contributing -1 when it is absent from the summed spikes and
        # +1 when it always spikes. Splitting each stage constant that way leaves a
        # stream constant of magnitude at most 1 for every legal band.
        stream_constant = (-band_lo, 1.0 + band_lo - band_hi, band_hi)
        # The accumulator width bounds how long a stage remembers an overshoot, so it
        # sets how fast a saturated stage tracks the input again. On a low-discrepancy
        # input stream, four integer bits emit the same spikes as any wider accumulator;
        # an input that delivers its spikes in long runs needs more width to hold the
        # overshoot, and 256 ones followed by 256 zeros on band (-1, 0) reads -0.4727 at
        # four integer bits against 0.0000 at ten.
        stage_config = {'polarity': 'bipolar', 'scale': 1, 'intwidth': 4, 'fracwidth': 0}
        #: Saturating adder whose low saturation places the ``lo`` floor.
        self.floor_stage = add_scale(stage_config)
        #: Saturating adder whose high saturation places the ``hi`` ceiling.
        self.ceiling_stage = add_scale(stage_config)
        #: Saturating adder that shifts the banded value back to the ``[lo, hi]`` band.
        self.shift_stage = add_scale(stage_config)

        # The stage constants ride Sobol streams because they are not the +/-1 an
        # entry supplies on its own. Addition is count based, so correlation between
        # a constant stream and the input stream leaves the sum unchanged; separate
        # dimensions keep the per-timestep spike counts spread instead.
        dim = config.get('dim', 2)
        # The Sobol generator accepts dimensions 1 through 21201 and the three stage
        # constants take dim, dim + 1, and dim + 2, so the first one stops at 21199.
        if type(dim) is not int or dim < 1 or dim > 21199:
            message = f'Invalid dim: <{dim}>; legal values: an integer in [1, 21199].'
            logger.error(message)
            raise AssertionError(message)
        #: Period of the precomputed band-constant spike sequences.
        self.constant_len = 2 ** (self.fracwidth + 1)
        #: Precomputed scalar band-constant spikes per stage, indexed by timestep.
        self.constant_bits = []
        for index, constant in enumerate(stream_constant):
            num_seq = gen_num_seq({'width': self.fracwidth + 1, 'generator': 'sobol', 'dim': dim + index})
            probability = torch.tensor((constant + 1.0) / 2.0, dtype=self.ntype)
            self.constant_bits.append([float(b) for b in torch.gt(probability, num_seq).type(self.stype).tolist()])
        #: Hardware latency and timing metadata for the composed clamp path.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Reset no class-owned state; :meth:`reset` resets the three stage adders.
        """
        pass


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
        index = (self.timestep_cur - 1) % self.constant_len
        # Each stage sums its input spike, its constant spike, and a fixed entry that
        # is absent in the floor and shift stages and always spikes in the ceiling stage.
        floor_out = self.floor_stage(input + self.constant_bits[0][index], entry=3, dim=None)
        ceiling_out = self.ceiling_stage(floor_out + self.constant_bits[1][index] + 1.0, entry=3, dim=None)
        return self.shift_stage(ceiling_out + self.constant_bits[2][index], entry=3, dim=None)

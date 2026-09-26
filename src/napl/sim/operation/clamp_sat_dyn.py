import torch

from napl.sim.base import napl_base
from .add_scale import add_scale


class clamp_sat_dyn(napl_base):
    r"""
    Clamp a rate-coded stream to a ``[lo, hi]`` band supplied at every timestep.

    Use this streaming operation when the band is a runtime input rather than a
    construction constant, as when a controller retunes the bounds while the
    stream runs. The target rate-domain operation for a call made with bounds
    :math:`\mathit{lo}` and :math:`\mathit{hi}` is

    .. math::

       y = \min(\max(x, \mathit{lo}), \mathit{hi}).

    Three chained bipolar saturating adders realize the band. ``lo`` and ``hi``
    are rate-coded spike streams the caller supplies alongside the input, so no
    band lives in the configuration and no bound is quantized to a grid here.
    The caller encodes them; ride each on a Sobol dimension distinct from the
    other bound and from the input. Because each stage adds by counting spikes,
    correlation between the two bound streams and the input leaves every stage
    sum unchanged, so separate dimensions only spread the per-timestep counts.

    The band is approximate: the output rate departs from the requested band,
    with the error reported by the printed output of the focused test
    ``tests/operation/test_clamp_sat_dyn.py``. The band the circuit places is the
    one the bound streams decode to over the run, so a bound stream whose rate
    moves mid-run moves the band with it, over the settling time of the stage
    accumulators. The stage accumulators are fixed at 4 integer bits, sized for
    low-discrepancy (Sobol-class) input and bound streams; a bursty or highly
    correlated stream exceeds that width, so a caller feeding one should expect
    the band to be placed less accurately.

    An inverted band, where the ``lo`` stream decodes above the ``hi`` stream,
    is a property of the streams rather than of the configuration and is not
    rejected. The ceiling wins and the floor is ignored, so the output follows
    the ``hi`` stream: a widely inverted band passes that stream through bit for
    bit, and a narrowly inverted one passes it with occasional spikes dropped.
    The output never carries a spike the ``hi`` stream does not, on any band, so
    an inverted band places the output rate at or below the ``hi`` rate.

    Unipolar and bipolar rate-coded inputs are supported; the input, both bound
    streams, and the output share the configured polarity. A unipolar band is
    clamped by the same bipolar circuit on the same spikes, which read as the
    bipolar value :math:`2p - 1` for the input and for both bounds alike. Three
    input streams are consumed and one output stream is produced.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import clamp_sat_dyn

        operation = clamp_sat_dyn({'polarity': 'bipolar'})
        output = operation(torch.tensor([0.0, 1.0]), torch.tensor(0.0), torch.tensor(1.0))

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
            }
        ):
        """
        Configure the stream polarity.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Encoding of the input, the two bound streams, and the output, either ``"unipolar"`` or ``"bipolar"``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity'], polarity_required=True)

        # Each stage is an add_scale of unit scale whose rate saturation clips its
        # bipolar output value to [-1, 1], so on an input of value v and bounds lo and
        # hi the stages compute
        #     u = sat(v - (1 + lo)),
        #     w = sat(u + (2 + lo - hi)),
        #     y = sat(w + (hi - 1)).
        # The first stage's low saturation lands at the floor, since v - (1 + lo) <= -1
        # holds exactly when v <= lo, so u = max(v, lo) - (1 + lo). The second stage's
        # high saturation lands at the ceiling, since u + (2 + lo - hi) = max(v, lo) +
        # 1 - hi >= 1 holds exactly when max(v, lo) >= hi, so w = min(max(v, lo), hi) +
        # 1 - hi. The second stage never saturates low and the third never saturates at
        # all when lo <= hi, because hi <= lo + 2 always holds for bounds in [-1, 1] and
        # the banded value already lies inside [-1, 1]. The third stage shifts the band
        # back, so y is the clamped value. With lo above hi the second stage saturates
        # high on every value, since max(v, lo) + 1 - hi >= lo + 1 - hi > 1, so w is
        # clipped to 1 and the third stage leaves the constant value hi.
        # A rail entry carries no stream: it contributes -1 when it never spikes and +1
        # when it always spikes. The bound streams enter directly for +lo and +hi and
        # inverted for -lo and -hi, since 1 - spike encodes the negated bipolar value.
        # The floor stage is then (v, rail -1, ~lo), the ceiling stage
        # (u, rail +1, rail +1, lo, ~hi), and the shift stage (w, rail -1, hi).
        # add_scale centers a bipolar sum by (entry - scale) / 2, which is 1 for the
        # three-entry stages and 2 for the five-entry ceiling stage. Both are whole raw
        # accumulator units at fracwidth 0, so the five-entry stage carries no half-unit
        # bias that the three-entry stages avoid.
        # The shift stage's accumulator never leaves [0, 1], so its width and fracwidth
        # are free of the input. The ceiling stage's two +1 rails hold its per-timestep
        # delta, u + lo + 1 - hi, in [0, 3], so its accumulator never goes negative and
        # ceiling_out = 0 requires that accumulator to be 0, which requires a zero delta,
        # which requires hi = 1. The shift stage's delta is ceiling_out + hi - 1, whose
        # only -1 case needs ceiling_out = 0 and hi = 0 together, which the previous line
        # forbids. So the shift delta is 0 or 1 for arbitrary bit sequences and the shift
        # accumulator stays in [0, 1], well inside the [-8, 7] rails four integer bits
        # gives.
        # The output is therefore ceiling_out and hi together on every band, so it never
        # carries a spike the hi stream does not. Dropping a hi spike needs ceiling_out = 0,
        # which needs both a zero ceiling delta, floor_out + lo = hi, and a ceiling
        # accumulator still at 0 entering the timestep, since that accumulator never goes
        # negative. Under an inverted band the ceiling delta is positive on most timesteps,
        # so the accumulator leaves 0 within the first few timesteps and does not return:
        # a narrowly inverted band drops hi spikes only over that warm-up, and a widely
        # inverted one, whose accumulator leaves 0 sooner still, passes the hi stream bit
        # for bit.
        # The accumulator width bounds how long a stage remembers an overshoot, so it
        # sets how fast a saturated stage tracks the input again. The five-entry ceiling
        # stage swings by up to +3 raw units per timestep against +2 for a three-entry
        # stage, yet four integer bits is still the narrowest width that emits the same
        # spikes as any wider accumulator when a static band and a low-discrepancy input
        # hold every stage near its rails: three integer bits departs, and the focused
        # test pins both ends. That equivalence is scoped to that regime and to no other.
        # A band that moves mid-run, and an input that steps between constant rates, both
        # leave a saturated stage holding an overshoot whose size the width bounds, so a
        # wider accumulator emits different spikes there; the focused test pins the spikes
        # this width emits in both of those regimes instead. A stream that delivers its
        # spikes in long runs needs more width to hold the overshoot.
        stage_config = {'polarity': 'bipolar', 'scale': 1, 'intwidth': 4, 'fracwidth': 0}
        #: Saturating adder whose low saturation places the ``lo`` floor.
        self.floor_stage = add_scale(stage_config)
        #: Saturating adder whose high saturation places the ``hi`` ceiling.
        self.ceiling_stage = add_scale(stage_config)
        #: Saturating adder that shifts the banded value back to the ``[lo, hi]`` band.
        self.shift_stage = add_scale(stage_config)
        #: Hardware latency and timing metadata for the composed clamp path.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'lo': 'rc', 'hi': 'rc', 'output': 'rc'}
        self.polarity_io = {
            'input': self.polarity, 'lo': self.polarity, 'hi': self.polarity,
            'output': self.polarity,
        }
        self.correlation_i = {}


    def _reset(self):
        """
        Reset no class-owned state; :meth:`reset` resets the three stage adders.
        """
        pass


    def forward(self, input, lo, hi):
        """
        Process one timestep of a rate-coded input stream against one timestep
        of each bound stream.

        Args:
            input: Tensor of current 0/1 input spikes.
            lo: Current 0/1 spikes of the lower-bound stream, broadcastable
                against ``input``.
            hi: Current 0/1 spikes of the upper-bound stream, broadcastable
                against ``input``.

        Returns:
            Output spike tensor with the broadcast shape of the three inputs,
            whose stream decodes to ``clamp(input, lo, hi)``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]), torch.tensor(0.0), torch.tensor(1.0))
        """
        floor_out = self.floor_stage(input + (1.0 - lo), entry=3, dim=None)
        ceiling_out = self.ceiling_stage(floor_out + 2.0 + lo + (1.0 - hi), entry=5, dim=None)
        return self.shift_stage(ceiling_out + hi, entry=3, dim=None)

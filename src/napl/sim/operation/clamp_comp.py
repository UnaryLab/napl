import math
import torch

from loguru import logger
from napl.sim.base import napl_base
from .encode import gen_num_seq
from .max import max
from .min import min


class clamp_comp(napl_base):
    r"""
    Clamp a rate-coded stream to a fixed ``[lo, hi]`` band with stream comparators.

    Use this streaming operation to bound a single rate-coded input to a
    configurable band. The target rate-domain operation is

    .. math::

       y = \min(\max(x, \mathit{lo}), \mathit{hi}).

    Two chained selectors realize the band: a maximum selector places the
    ``lo`` floor and a minimum selector places the ``hi`` ceiling on its
    result. Each selector compares the two streams it is given and routes one
    of them to its output, so the output spikes are always input or bound
    spikes and no accumulator carries an overshoot. The legal band is the whole
    value range of the polarity: ``lo`` and ``hi`` both inside the range, with
    ``lo`` strictly below ``hi``. The two bounds are encoded inside the
    operation, each onto its own Sobol dimension.

    The band is approximate: the output rate departs from the requested band by
    an amount that varies across the legal band box, with the worst case over
    that whole box reported by the printed output of the focused test
    ``tests/operation/test_clamp_comp.py``.

    Unipolar and bipolar rate-coded inputs are supported; the input and the
    output share the configured polarity, and both bounds are encoded in it.
    Selection compares spike counts, and the bipolar value :math:`2p - 1` grows
    with the rate :math:`p`, so the same comparison orders both polarities and
    no band mapping is applied. One input stream is consumed and one output
    stream is produced.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import clamp_comp

        operation = clamp_comp({'polarity': 'bipolar', 'lo': -0.5, 'hi': 0.5})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """
    #: Dominant hardware mechanism of this class.
    mechanism = 'finite-state-machine'


    def __init__(
            self,
            config={
                'polarity': 'bipolar',
                'lo': -0.5,
                'hi': 0.5,
            }
        ):
        """
        Configure the polarity, the band, and the bound-stream dimensions.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, either ``"unipolar"`` or ``"bipolar"``.
              - **lo**: Lower band bound, a number inside the polarity's legal value range and below **hi**.
              - **hi**: Upper band bound, a number inside the polarity's legal value range and above **lo**.
              - **dim**: One-based Sobol dimension of the ``lo`` bound stream, with the ``hi`` stream on ``dim + 1``; the default is ``2``, and the largest legal value is ``21200``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'lo', 'hi'], optional_key_list=['dim'], polarity_required=True)

        # Bipolar rate coding spans values [-1, 1]; unipolar spans [0, 1].
        legal_low = -1.0 if self.polarity == 'bipolar' else 0.0
        legal_high = 1.0
        lo = config['lo']
        hi = config['hi']
        for label, bound in (('lo', lo), ('hi', hi)):
            # The comparisons below are both False for a NaN bound, so the finiteness
            # test runs before either.
            if (type(bound) not in (int, float) or not math.isfinite(bound)
                    or bound < legal_low or bound > legal_high):
                message = (f'Invalid {label}: <{bound}>; legal values: a number in '
                           f'[{legal_low}, {legal_high}] for {self.polarity} polarity.')
                logger.error(message)
                raise AssertionError(message)
        #: Lower band bound.
        self.lo = float(lo)
        #: Upper band bound.
        self.hi = float(hi)
        if self.lo >= self.hi:
            message = f'Invalid band: lo <{self.lo}> must be strictly below hi <{self.hi}>.'
            logger.error(message)
            raise AssertionError(message)

        dim = config.get('dim', 2)
        # The Sobol generator accepts dimensions 1 through 21201 and the two bound
        # streams take dim and dim + 1, so the first one stops at 21200.
        if type(dim) is not int or dim < 1 or dim > 21200:
            message = f'Invalid dim: <{dim}>; legal values: an integer in [1, 21200].'
            logger.error(message)
            raise AssertionError(message)

        #: Period of the precomputed bound spike sequences.
        self.constant_len = 2 ** 9
        # Each selector compares its two streams by spike count, so a bound is placed
        # to the rate its own stream carries: a 512-long sequence resolves a bound to
        # 1/512. The two bounds ride separate dimensions, which keeps the timestep a
        # bound spikes on independent of the other bound.
        #: Precomputed scalar ``lo`` bound spikes, indexed by timestep.
        self.lo_bits: torch.Tensor
        #: Precomputed scalar ``hi`` bound spikes, indexed by timestep.
        self.hi_bits: torch.Tensor
        for name, bound, index in (('lo_bits', self.lo, 0), ('hi_bits', self.hi, 1)):
            num_seq = gen_num_seq({'width': 9, 'generator': 'sobol', 'dim': dim + index})
            # A unipolar rate p and the bipolar value 2p - 1 threshold against the same
            # sequence, so a bipolar bound moves to the rate that denotes it.
            probability = bound if self.polarity == 'unipolar' else (bound + 1.0) / 2.0
            self.register_buffer(
                name,
                torch.gt(torch.tensor(probability, dtype=self.ntype), num_seq).type(self.stype),
            )

        #: Selector that places the ``lo`` floor on the input stream.
        self.floor_stage = max()
        #: Selector that places the ``hi`` ceiling on the floored stream.
        self.ceiling_stage = min()
        #: Hardware latency and timing metadata for the composed clamp path.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Reset no class-owned state; :meth:`reset` resets the two selectors.
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
        # The bound bits are 0-dim, and the selectors hold their decision at the rank of the
        # first call, so a 0-dim input is carried as a one-element tensor and the output carries
        # that rank.
        input = torch.atleast_1d(input)

        index = (self.timestep_cur - 1) % self.constant_len
        floor_out, _ = self.floor_stage(input, self.lo_bits[index])
        output, _ = self.ceiling_stage(floor_out, self.hi_bits[index])
        return output

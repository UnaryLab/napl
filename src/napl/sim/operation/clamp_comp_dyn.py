import torch

from napl.sim.base import napl_base
from .max import max
from .min import min


class clamp_comp_dyn(napl_base):
    r"""
    Clamp a rate-coded stream to a ``[lo, hi]`` band supplied at every timestep.

    Use this streaming operation when the band is a runtime input rather than a
    construction constant, as when a controller retunes the bounds while the
    stream runs. The target rate-domain operation for a call made with bounds
    :math:`\mathit{lo}` and :math:`\mathit{hi}` is

    .. math::

       y = \min(\max(x, \mathit{lo}), \mathit{hi}).

    Two chained selectors realize the band: a maximum selector places the
    ``lo`` floor and a minimum selector places the ``hi`` ceiling on its
    result. ``lo`` and ``hi`` are rate-coded spike streams the caller supplies
    alongside the input, so no band lives in the configuration. The caller
    encodes them; ride each on a Sobol dimension distinct from the other bound
    and from the input. The band the circuit places is the one the bound
    streams decode to over the run, so a bound stream whose rate moves mid-run
    moves the band with it.

    The band is approximate: the output rate departs from the requested band,
    with the error reported by the printed output of the focused test
    ``tests/operation/test_clamp_comp_dyn.py``.

    An inverted band, where the ``lo`` stream decodes above the ``hi`` stream,
    is a property of the streams rather than of the configuration and is not
    rejected. The ceiling wins and the floor is ignored, so the output rate
    follows the ``hi`` rate. The output is not the ``hi`` stream masked: the
    ceiling selector routes whole spikes from whichever of its two streams its
    latched decision names, so it can carry spikes the ``hi`` stream does not as
    well as drop spikes it does. Which spikes it carries follows the rates of
    the input and the two bound streams, not the polarity those streams are read
    in: the same rates read in either polarity give the same output stream.

    Unipolar and bipolar rate-coded inputs are supported; the input, both bound
    streams, and the output share the configured polarity. Selection compares
    spike counts, and the bipolar value :math:`2p - 1` grows with the rate
    :math:`p`, so the same comparison orders both polarities and no band
    mapping is applied. Three input streams are consumed and one output stream
    is produced.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import clamp_comp_dyn

        operation = clamp_comp_dyn({'polarity': 'bipolar'})
        output = operation(torch.tensor([0.0, 1.0]), torch.tensor(0.0), torch.tensor(1.0))

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

        #: Selector that places the ``lo`` floor on the input stream.
        self.floor_stage = max()
        #: Selector that places the ``hi`` ceiling on the floored stream.
        self.ceiling_stage = min()
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
        Reset no class-owned state; :meth:`reset` resets the two selectors.
        """
        pass


    def forward(self, input, lo, hi):
        """
        Process one timestep of the input stream and both bound streams.

        Args:
            input: Tensor of current 0/1 input spikes.
            lo: Current 0/1 spikes of the lower-bound stream.
            hi: Current 0/1 spikes of the upper-bound stream.

        Returns:
            Output spike tensor with the same shape as ``input``, whose stream
            decodes to ``clamp(input, lo, hi)`` for the bands the two bound
            streams decode to.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]), torch.tensor(0.0), torch.tensor(1.0))
        """
        # The selectors hold their decision at the rank of the first call, so a 0-dim input is
        # carried as a one-element tensor and the output carries that rank.
        input = torch.atleast_1d(input)

        floor_out, _ = self.floor_stage(input, lo)
        output, _ = self.ceiling_stage(floor_out, hi)
        return output

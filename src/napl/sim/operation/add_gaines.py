import torch
import math

from napl.sim.base import napl_base
from .encode import encode
from loguru import logger


class add_gaines(napl_base):
    """
    Add rate-coded spike streams with the Gaines MUX or OR construction.

    Use scaled mode to estimate the mean of a power-of-two number of unipolar
    or bipolar streams. Use non-scaled mode for an OR-based approximation to a
    small unipolar sum.

    The target reductions are

    .. math::

       y_{\\mathrm{scaled}} = \\frac{1}{n}\\sum_i x_i,\\qquad
       y_{\\mathrm{non-scaled}} = \\min\\left(1,\\sum_i x_i\\right).

    Scaled mode holds the number sequence used to pick one input stream per
    timestep, so it needs **entry** and **generator** and accepts a power-of-two
    input count only. Non-scaled mode is unipolar only, and its output
    approaches the clipped sum only while the input streams do not overlap.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import add_gaines

        adder = add_gaines({'polarity': 'unipolar', 'scaled': True,
                            'entry': 2, 'generator': 'Sobol', 'dim': 1})
        output = adder(torch.tensor([1, 0], dtype=torch.int8), dim=0)

    .. container:: api-references

        .. rubric:: References

        *Stochastic Computing Systems*, Advances in Information Systems Science, 1969.
    """
    #: The MUX select sequence is encoded from a held number sequence, so the
    #: RTL counterpart holds its own encoder instead of sharing an external one.
    internal_encode = True


    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
            'scaled' : True,
            'entry' : 8,
            'generator' : 'Sobol',
            'dim' : 1,
        }
    ):
        """
        Select scaled MUX addition or non-scaled OR addition.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **scaled**: Use MUX-based scaled addition when ``True`` or unipolar OR addition when ``False``; the default is ``True``.
              - **entry**: Number of inputs in scaled mode. It must be a power of two; the default is ``8``.
              - **generator**: Number-sequence generator used for MUX selection; the default is ``"Sobol"``.
              - **dim**: Generator dimension forwarded when the selection sequence is built; the default is ``1``.
              - **seed**: Optional LFSR seed used when **generator** is ``"lfsr"``; the default is ``None``.
              - **taps**: Optional LFSR feedback taps used when **generator** is ``"lfsr"``; the default is ``None``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'scaled'], optional_key_list=['entry', 'generator', 'dim', 'seed', 'taps'], polarity_required=True)

        #: Whether addition uses MUX-based averaging instead of an OR reduction.
        self.scaled = config['scaled']
        if self.polarity == 'bipolar' and not self.scaled:
            message = 'Non-scaled Gaines addition does not support bipolar data.'
            logger.error(message)
            raise AssertionError(message)

        if self.scaled:
            if 'entry' not in config or 'generator' not in config:
                message = 'Scaled Gaines addition requires <entry> and <generator> in configuration.'
                logger.error(message)
                raise AssertionError(message)
            #: Number of input streams accepted by the scaled MUX adder.
            self.entry = config['entry']
            if math.log2(self.entry) != math.ceil(math.log2(self.entry)):
                message = f'Input entry <{self.entry}> is not a power of 2.'
                logger.error(message)
                raise AssertionError(message)
            # The encoder supplies the sequence once and is not retained.
            reference_encode = encode({'polarity': 'unipolar',
                                       'timestep': self.entry,
                                       'generator': config['generator'],
                                       'dim': config.get('dim', 1),
                                       'seed': config.get('seed', None),
                                       'taps': config.get('taps', None)})
            # Python scalar indices avoid device synchronization on each timestep.
            #: Periodic input indices selected by the configured number sequence.
            self.sel_seq = reference_encode.num_seq.mul(self.entry).type(torch.long).tolist()
        #: Hardware latency and timing metadata for the Gaines adder.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Reset no local state; the kernel has no class-owned mutable state.
        """
        pass


    def forward(self, input: torch.Tensor, dim: int = 0):
        """
        Reduce one timestep of input spikes.

        Args:
            input: Spike tensor containing the streams to add.
            dim: Dimension containing the input streams; the default is ``0``.

        Returns:
            The selected spike in scaled mode or the elementwise OR reduction
            in non-scaled mode. Scaled mode indexes the selection sequence by
            ``timestep_cur``.

        **Example:**

        .. code-block:: python

            output = adder(torch.tensor([1, 0], dtype=torch.int8), dim=0)
        """
        if self.scaled:
            if input.size(dim) != self.entry:
                message = f'Input size <{input.size(dim)}> along dim <{dim}> != configured entry <{self.entry}>.'
                logger.error(message)
                raise AssertionError(message)
            output = input.select(dim, self.sel_seq[(self.timestep_cur - 1) % len(self.sel_seq)])
        else:
            # For 0/1 spikes, max reduction is unipolar OR and preserves stype.
            output = torch.amax(input, dim)
        return output.type(self.stype)

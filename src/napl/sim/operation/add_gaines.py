import torch
import math

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import encode
from loguru import logger


class add_gaines(napl_base):
    """
    Add rate-coded spike streams with the Gaines MUX or OR construction.

    Use scaled mode to estimate the mean of a power-of-two number of unipolar
    or bipolar streams. Use non-scaled mode for an OR-based approximation to a
    small unipolar sum.

    The precise target reductions are

    .. math::

       y_{\\mathrm{scaled}} = \\frac{1}{n}\\sum_i x_i,\\qquad
       y_{\\mathrm{non-scaled}} = \\min\\left(1,\\sum_i x_i\\right).

    Scaled addition quantizes the number sequence into an input index rather than
    comparing a value against it, so a :class:`napl.encode` instance supplies the
    sequence at construction and is not retained. The selection position is held
    locally.

    In scaled mode, with ``entry`` inputs and the configured number sequence
    ``G_t``, the exact selected-input reduction is

    .. math::

       j_t = \\lfloor \\mathit{entry} G_t \\rfloor, \\qquad
       y_t = x_{j_t,t}.

    In non-scaled mode the exact unipolar reduction is

    .. math::

       y_t = \\max_i x_{i,t}.

    Non-scaled bipolar input is not supported.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import add_gaines

        adder = add_gaines({'polarity': 'unipolar', 'scaled': True,
                            'entry': 2, 'generator': 'Sobol', 'dim': 1})
        output = adder(torch.tensor([1, 0], dtype=torch.int8), dim=0)

    .. container:: api-references

        .. rubric:: References

        B. R. Gaines, *Stochastic Computing Systems*, Advances in Information Systems Science, vol. 2, 1969.
    """


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
        super().__init__(config, ['polarity', 'scaled'], polarity_required=True)

        #: Whether addition uses MUX-based averaging instead of an OR reduction.
        self.scaled = config['scaled']
        assert not (self.polarity == 'bipolar' and not self.scaled), \
            logger.error('Non-scaled Gaines addition does not support bipolar data.')

        if self.scaled:
            assert 'entry' in config and 'generator' in config, \
                logger.error('Scaled Gaines addition requires <entry> and <generator> in configuration.')
            #: Number of input streams accepted by the scaled MUX adder.
            self.entry = config['entry']
            assert math.log2(self.entry) == math.ceil(math.log2(self.entry)), \
                logger.error(f'Input entry <{self.entry}> is not a power of 2.')
            # The encoder supplies the sequence once and is not retained.
            reference_encode = encode({'polarity': 'unipolar',
                                       'timestep': self.entry,
                                       'generator': config['generator'],
                                       'dim': config.get('dim', 1),
                                       'seed': config.get('seed', None),
                                       'taps': config.get('taps', None)})
            # Python scalar indices avoid device synchronization on each timestep.
            #: Periodic input indices selected by the configured number sequence.
            self.sel_seq = torch.floor(
                reference_encode.num_seq.mul(self.entry)).type(torch.long).tolist()
            #: Current position in :attr:`sel_seq`, advanced after each scaled call.
            self.idx = 0
        #: Hardware latency and timing metadata for the Gaines adder.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restart the local MUX selection sequence at its first value.
        """
        self.idx = 0


    def forward(self, input: torch.Tensor, dim: int = 0):
        """
        Reduce one timestep of input spikes.

        Args:
            input: Spike tensor containing the streams to add.
            dim: Dimension containing the input streams; the default is ``0``.

        Returns:
            The selected spike in scaled mode or the elementwise OR reduction
            in non-scaled mode. Scaled calls advance the selection sequence.

        **Example:**

        .. code-block:: python

            output = adder(torch.tensor([1, 0], dtype=torch.int8), dim=0)
        """
        if self.scaled:
            assert input.size(dim) == self.entry, \
                logger.error(f'Input size <{input.size(dim)}> along dim <{dim}> != configured entry <{self.entry}>.')
            output = input.select(dim, self.sel_seq[self.idx])
            self.idx = (self.idx + 1) % len(self.sel_seq)
        else:
            # For 0/1 spikes, max reduction is unipolar OR and preserves stype.
            output = torch.amax(input, dim)
        return output.type(self.stype)

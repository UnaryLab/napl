import torch
import math

from napl.sim.base import napl_base, hw_params
from napl.sim.module import gen_num_seq
from loguru import logger


class add_gaines(napl_base):
    """
    Add rate-coded spike streams with the Gaines MUX or OR construction.

    Use scaled mode to estimate the mean of a power-of-two number of unipolar
    or bipolar streams. Use non-scaled mode for an OR-based approximation to a
    small unipolar sum.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import add_gaines

        adder = add_gaines({'polarity': 'unipolar', 'scaled': True,
                            'entry': 2, 'generator': 'Sobol', 'dim': 1})
        output = adder(torch.tensor([1, 0], dtype=torch.int8), dim=0)

    .. container:: api-references

        .. rubric:: References

        B. R. Gaines, *Stochastic Computing Systems*, 1969.
    """
    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
            # whether the sum is scaled by the entry count (MUX); False uses an OR gate
            'scaled' : True,
            # number of input streams along the reduce dimension; power of 2 (MUX select width)
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
        # combinational MUX / OR gate: no registers
        self.hw = hw_params(pp_delay=0)

        self.scaled = config['scaled']
        assert not (self.polarity == 'bipolar' and not self.scaled), \
            logger.error('Non-scaled Gaines addition does not support bipolar data.')

        if self.scaled:
            assert 'entry' in config and 'generator' in config, \
                logger.error('Scaled Gaines addition requires <entry> and <generator> in configuration.')
            self.entry = config['entry']
            assert math.log2(self.entry) == math.ceil(math.log2(self.entry)), \
                logger.error(f'Input entry <{self.entry}> is not a power of 2.')
            config['width'] = int(math.log2(self.entry))
            # MUX select sequence scaled to integers in [0, entry); static python list so the
            # per-timestep index is a python scalar (no device-scalar fetch per timestep)
            self.sel_seq = torch.floor(gen_num_seq(config).mul(self.entry)).type(torch.long).tolist()
            # index of numbers in the select seq
            self.idx = 0


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
        # input is a spike tensor; reduce over `dim`
        if self.scaled:
            assert input.size(dim) == self.entry, \
                logger.error(f'Input size <{input.size(dim)}> along dim <{dim}> != configured entry <{self.entry}>.')
            # MUX: forward the selected stream for both polarities
            output = input.select(dim, self.sel_seq[self.idx])
            self.idx = (self.idx + 1) % len(self.sel_seq)
        else:
            # OR gate over the inputs, unipolar only; max of 0/1 spikes == OR,
            # stays in stype (no float sum round-trip)
            output = torch.amax(input, dim)
        return output.type(self.stype)

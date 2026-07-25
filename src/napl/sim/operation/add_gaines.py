import torch
import math

from napl.sim.base import napl_base, hw_params
from napl.sim.module import gen_num_seq
from loguru import logger


class add_gaines(napl_base):
    """
    Gaines addition of `entry` spike streams along a dimension.
    1) scaled: a MUX picks one input stream per timestep via an RNG select,
       computing sum/entry (unipolar and bipolar).
    2) non-scaled: an OR gate over the inputs approximates the sum for small
       values (unipolar only).
    Reference:
    1) B. R. Gaines, 'Stochastic Computing Systems', 1969
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
        self.idx = 0


    def forward(self, input: torch.Tensor, dim: int = 0):
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

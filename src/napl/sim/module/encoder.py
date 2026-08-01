import torch
import math
import numpy as np

from napl.utils import *
from napl.sim.base import napl_base, global_config
from loguru import logger
from pylfsr import LFSR



def get_lfsr_seq(width=8, seed:int=None, taps:list=None) -> torch.tensor:
    """
    return a lfsr sequence of length 2**width within [0, 1]
    """
    if seed is None:
        # The default seed makes generated sequences reproducible.
        seed = [0 for _ in range(width-1)] + [1]
    else:
        seed = int(seed) % (2**width)
        seed = [int(x) for x in np.binary_repr(seed, width=width)]

    if taps is None:
        polylist = LFSR().get_fpolyList(m=width)
        poly = polylist[0]
    else:
        assert isinstance(taps, list) and len(taps) > 0, \
            f'Error: the input taps {taps} needs to be a non-empty list.'
        poly = taps

    L = LFSR(fpoly=poly,initstate =seed)

    lfsr_seq = []
    for i in range(2**width):
        lfsr_seq.append(int(''.join(map(str, L.state)), 2)/2**width)
        L.next()

    return torch.tensor(lfsr_seq, dtype=global_config.ntype)


def get_sysrand_seq(width=8):
    """
    return a system random sequence of length 2**width within [0, 1]
    """
    return torch.randperm(2**width) / 2**width


def gen_num_seq(config={
            'width' : 8,
            'generator' : 'Sobol'
        }):
    """
    Return a number sequence of size 2**width, with each number being a number with [0, 1].
    """

    width = config['width']
    generator = config['generator'].lower()
    seq_len = 2**width

    legal_rngs = ['sobol', 'lfsr', 'sys', 'rc', 'tc', 'rate', 'temporal']

    assert generator in legal_rngs, \
        logger.error(f'Invalid sequence generator: <{generator}>; legal values: <{legal_rngs}>.')

    if (generator == 'sobol') or (generator == 'rc') or (generator == 'rate'):
        # Rate coding uses the requested Sobol dimension.
        dim = config.get('dim', 1)
        num_seq = torch.quasirandom.SobolEngine(dim).draw(seq_len)[:, dim-1].view(seq_len)
    elif (generator == 'tc') or (generator == 'temporal'):
        # Descending thresholds make temporal streams emit ones before zeros.
        num_seq = torch.tensor([x/seq_len for x in range(seq_len-1, -1, -1)])
    elif generator == 'lfsr':
        num_seq = get_lfsr_seq(width=width, seed=config.get('seed', None), taps=config.get('taps', None))
    elif generator == 'sys':
        num_seq = get_sysrand_seq(width=width)

    return num_seq.type(global_config.ntype)


def input_scale(input, quantile=1):
    """
    Scale input data to [-1, 1] in a symmetric manner, which meets bipolar/unipolar requirements.
    The remaining data count for 'quantile' quantile of the total data.
    The input quantile needs to be within (0, 1].
    """

    assert quantile > 0 and quantile <= 1, \
        logger.error(f'Invalid quantile: <{quantile}>; legal values: (0, 1].')

    quantile_lower = 0.5 - quantile / 2
    quantile_upper = 0.5 + quantile / 2

    lower_bound = torch.quantile(input, quantile_lower)
    upper_bound = torch.quantile(input, quantile_upper)
    scale = torch.max(lower_bound.abs(), upper_bound.abs())
    output = input.clamp(lower_bound, upper_bound).div(scale)
    return output


class encoder(napl_base):
    """Encode numeric values as a unary spike stream.

    Use this module to generate one rate-coded or temporal-coded spike tensor per
    call. Values are interpreted in ``[0, 1]`` for unipolar encoding and
    ``[-1, 1]`` for bipolar encoding.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import encoder

        enc = encoder({"polarity": "unipolar", "timestep": 4,
                       "generator": "sobol"})
        output_spike = enc(torch.tensor([0.25, 0.75]))
    """

    def __init__(
            self,
            config:dict={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
                }
        ):
        """Configure the stream length and number-sequence generator.

        Args:
            config: Configuration mapping with these keys:

                * **polarity** - ``"unipolar"`` or ``"bipolar"``. Defaults to
                  ``"bipolar"``.
                * **timestep** - Requested positive stream length. Defaults to
                  ``256``; the generated sequence period is the next power of two.
                * **generator** - ``"sobol"``, ``"lfsr"``, ``"sys"``,
                  ``"rc"``, ``"tc"``, ``"rate"``, or ``"temporal"``.
                  Defaults to ``"sobol"``.
                * **dim** - One-based Sobol dimension. Defaults to ``1``.
                * **seed** - Optional integer LFSR seed. Defaults to ``None``.
                * **taps** - Optional non-empty LFSR feedback-tap list. Defaults
                  to ``None``.
                * **name** - Optional instance label. Defaults to ``None``.

        Construction generates and stores the complete number sequence.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)

        #: Requested number of output-spike timesteps in the stream.
        self.timestep = config['timestep']
        assert self.timestep > 0, logger.error(f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        #: Bit width of the power-of-two number-sequence period.
        self.width = math.ceil(math.log2(self.timestep))
        #: Lowercase name of the configured number-sequence generator.
        self.generator = config['generator'].lower()
        #: Number of thresholds in the generated periodic sequence.
        self.len = 2**self.width

        self._is_bipolar = (self.polarity == 'bipolar')

        config_updated = {'width': self.width}
        config_updated.update(config)
        #: Complete threshold sequence used to encode successive timesteps.
        self.num_seq: torch.Tensor
        self.register_buffer('num_seq', gen_num_seq(config=config_updated))

        self._prob_cache = None


    def _reset(self):
        """Drop the cached bipolar probability transform.

        The number sequence is unchanged. This hook returns ``None`` and is
        called by ``reset()``, which separately resets the timestep.
        """
        self._prob_cache = None


    def forward(self, input: torch.Tensor):
        """Encode one timestep for a numeric input tensor.

        Args:
            input: Numeric tensor in ``[0, 1]`` for unipolar encoding or
                ``[-1, 1]`` for bipolar encoding.

        Returns:
            A same-shaped ``0``/``1`` spike tensor using the configured global
            spike dtype.

        Calling the module advances ``timestep_cur`` and selects the next number
        in the periodic sequence. The input tensor is not modified.
        """
        if self._is_bipolar:
            c = self._prob_cache
            if c is not None and c[0] is input and c[1] == input._version:
                prob = c[2]
            else:
                prob = (input + 1)/2
                if not input.requires_grad:
                    self._prob_cache = (input, input._version, prob)
        else:
            prob = input
        spike = torch.gt(prob, self.num_seq[(self.timestep_cur-1) % self.len]).type(self.stype)
        return spike

import torch, math
import numpy as np

from napl.utils import *
from napl.base import napl_base, global_config
from loguru import logger
from pylfsr import LFSR



def get_lfsr_seq(width=8, seed:int=None, taps:list=None) -> torch.tensor:
    """
    return a lfsr sequence of length 2**width within [0, 1]
    """
    if seed is None:
        # always start from a fixed seed if no seed is provided
        # this is to ensure the same sequence is generated every time
        seed = [0 for _ in range(width-1)] + [1]
    else:
        # input seed is a integer, convert it to a list of binary bits
        # modulo the seed to the width
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
    # return torch.randperm(2**width)/2**width
    return torch.rand(2**width)
    
    
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
        # get the requested dimension of sobol random number sequence
        # rate coding defaults to sobol sequence
        dim = config.get('dim', 1)
        num_seq = torch.quasirandom.SobolEngine(dim).draw(seq_len)[:, dim-1].view(seq_len)
    elif (generator == 'tc') or (generator == 'temporal'):
        # temporal coding defaults to ascending counter sequence
        # the output sequence is in an ascending order
        # the temporal coding starts with 0s, followed by 1s
        num_seq = torch.tensor([x/seq_len for x in range(0, seq_len)])
    elif generator == 'lfsr':
        num_seq = get_lfsr_seq(width=width, seed=config.get('seed', None), taps=config.get('taps', None))
    elif generator == 'sys':
        num_seq = get_sysrand_seq(width=width)
        
    return torch.nn.Parameter(num_seq.type(global_config.ntype), requires_grad=False)
    

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
    def __init__(
            self, 
            config:dict={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
                }
        ):
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)

        self.timestep = config['timestep']
        assert self.timestep > 0, logger.error(f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        self.width = math.ceil(math.log2(self.timestep))
        self.generator = config['generator'].lower()
        self.len = 2**self.width

        # generate the number sequence
        # the sequence is used to compare with the input data
        config_updated = {'width': self.width}
        config_updated.update(config)
        self.num_seq = gen_num_seq(config=config_updated)
        

    def reset(self, verbose=False):
        """
        Reset the timestep and spike count.
        """
        self.timestep_cur = 0
        

    def forward(self, input: torch.Tensor):
        self.tick()
        # use gt to generate the spike
        # if input is 0, then a all 0 spike train is generated
        # if input is 1, then one spike in the spike train will be 0
        if self.polarity == 'bipolar':
            prob = (input + 1)/2
        elif self.polarity == 'unipolar':
            prob = input
        spike = torch.gt(prob, self.num_seq[(self.timestep_cur-1) % self.len]).type(self.stype)
        return spike
        

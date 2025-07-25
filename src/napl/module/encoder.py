import torch, math
import numpy as np

from napl.utils import *
from napl.base import napl_base, global_config
from loguru import logger
from pylfsr import LFSR



def get_lfsr_seq(bitwidth=8, seed:int=None, taps:list=None) -> torch.tensor:
    """
    return a lfsr sequence of length 2**bitwidth within [0, 1]
    """
    if seed is None:
        # always start from a fixed seed if no seed is provided
        # this is to ensure the same sequence is generated every time
        seed = [0 for _ in range(bitwidth-1)] + [1]
    else:
        # input seed is a integer, convert it to a list of binary bits
        # modulo the seed to the bitwidth
        seed = int(seed) % (2**bitwidth)
        seed = [int(x) for x in np.binary_repr(seed, bitwidth=bitwidth)]

    if taps is None:
        polylist = LFSR().get_fpolyList(m=bitwidth)
        poly = polylist[0]
    else:
        assert isinstance(taps, list) and len(taps) > 0, \
            f'Error: the input taps {taps} needs to be a non-empty list.'
        poly = taps

    L = LFSR(fpoly=poly,initstate =seed)

    lfsr_seq = []
    for i in range(2**bitwidth):
        lfsr_seq.append(int(''.join(map(str, L.state)), 2)/2**bitwidth)
        L.next()

    return torch.tensor(lfsr_seq, dtype=global_config.ntype)


def get_sysrand_seq(bitwidth=8):
    """
    return a system random sequence of length 2**bitwidth within [0, 1]
    """
    # return torch.randperm(2**bitwidth)/2**bitwidth
    return torch.rand(2**bitwidth)
    
    
def gen_num_seq(config={
            'bitwidth' : 8, 
            'generator' : 'Sobol'
        }):
    """
    Return a number sequence of size 2**bitwidth, with each number being a number with [0, 1].
    """

    bitwidth = config['bitwidth']
    generator = config['generator'].lower()
    seq_len = 2**bitwidth

    legal_rngs = ['sobol', 'lfsr', 'sys', 'rc', 'tc', 'tc_asc', 'tc_dec', 'tc01', 'tc10']

    assert generator in ['sobol', 'race', 'lfsr', 'sys', 'rc', 'tc', 'race10', 'tc10'], \
        logger.error(f'Invalid sequence generator: <{generator}>; legal values: <{legal_rngs}>.')
    
    if (generator == 'sobol') or (generator == 'rc'):
        # get the requested dimension of sobol random number sequence
        # rate coding defaults to sobol sequence
        dim = config.get('dim', 1)
        num_seq = torch.quasirandom.SobolEngine(dim).draw(seq_len)[:, dim-1].view(seq_len)
    elif (generator == 'tc') or (generator == 'tc_asc') or (generator == 'tc01'):
        # temporal coding defaults to ascending counter sequence
        # the output sequence is in an ascending order
        num_seq = torch.tensor([x/seq_len for x in range(0, seq_len)])
    elif (generator == 'tc_dec') or (generator == 'tc10'):
        # the output sequence is in a descending order
        num_seq = torch.tensor([x/seq_len for x in range(seq_len-1, -1, -1)])
    elif generator == 'lfsr':
        num_seq = get_lfsr_seq(bitwidth=bitwidth, seed=config.get('seed', None), taps=config.get('taps', None))
    elif generator == 'sys':
        num_seq = get_sysrand_seq(bitwidth=bitwidth)
        
    return torch.nn.Parameter(num_seq.type(global_config.ntype), requires_grad=False)
    

def input_scale(input, quantile=1):
    """
    Scale input data to [-1, 1] in a symmetric manner, which meets bipolar/unipolar bitstream requirements.
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
                'mode': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
                }
        ):
        super().__init__()

        # check config
        check_config(config, ['mode', 'timestep', 'generator'])
        self.mode = check_mode(config)
        self.name = check_name(config)

        self.timestep = config['timestep']
        assert self.timestep > 0, logger.error(f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        self.bitwidth = math.ceil(math.log2(self.timestep))
        self.generator = config['generator'].lower()
        self.len = 2**self.bitwidth

        # generate the number sequence
        # the sequence is used to compare with the input data
        config_updated = {'bitwidth': self.bitwidth}
        config_updated.update(config)
        self.num_seq = gen_num_seq(config=config_updated)
        
        self.timestep_cur = 0


    def reset(self):
        """
        Reset the timestep and spike count.
        """
        self.timestep_cur = 0
        

    def forward(self, input: torch.Tensor):
        # use gt to generate the spike
        # if input is 0, then a all 0 spike train is generated
        # if input is 1, then one spike in the spike train will be 0
        self.timestep_cur = self.timestep_cur % self.len
        if self.mode == 'bipolar':
            prob = (input + 1)/2
        elif self.mode == 'unipolar':
            prob = input
        spike = torch.gt(prob, self.num_seq[self.timestep_cur]).type(self.stype)
        self.timestep_cur += 1
        return spike
        

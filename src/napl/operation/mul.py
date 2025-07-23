import torch, math

from napl.utils import *
from napl.base import napl_base
from napl.module.encoder import gen_num_seq
from loguru import logger


class mul_csg(napl_base):
    """
    This module is for unary multiplication with conditinal spike generation, supporting unipolar/bipolar.
    References:
    1) uGEMM: Unary Computing Architecture for GEMM Applications
    2) uGEMM: Unary Computing for GEMM Applications
    """
    def __init__(
            self,
            config={
                'mode': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
            }
        ):
        super().__init__()

        check_config(config, ['mode', 'timestep', 'generator'])

        # data representation
        self.mode = config['mode'].lower()
        check_mode(self.mode)
        
        self.timestep = config['timestep']
        assert self.timestep > 0, logger.error(f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        self.bitwidth = math.ceil(math.log2(self.timestep))
        self.generator = config['generator'].lower()
        self.len = 2**self.bitwidth

        # generate the number sequence
        # the sequence is used to compare with the input data
        # this is ntype tensor
        self.num_seq = gen_num_seq(config={'bitwidth': self.bitwidth, 
                                        'generator': self.generator})
        
        # seq_idx is used later as an enable signal, get update every cycled
        self.seq_idx = torch.nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)
        # Generate two seperate spike generators and two enable signals for bipolar mode
        if self.mode == "bipolar":
            self.seq_idx_inv = torch.nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)
        
        # only compute the prob of in_1 once in the first call
        self.in_1_prob = None
        self.is_first_call = True

    
    def reset(self):
        """
        Reset the seq_idx and seq_idx_inv.
        """
        self.seq_idx.data = torch.zeros(1, dtype=torch.long)
        if self.mode == "bipolar":
            self.seq_idx_inv.data = torch.zeros(1, dtype=torch.long)
        self.in_1_prob = None
        self.is_first_call = True

    
    def forward(self, in_0, in_1: torch.tensor=None):
        if self.is_first_call is True:
            assert in_1 is not None, logger.error('in_1 is None, please provide a valid in_1 tensor.')
            self.in_1_prob = (in_1 + 1) / 2 if self.mode == 'bipolar' else in_1
            self.is_first_call = False

        # generate the conditional spike
        spike_csg = torch.gt(self.in_1_prob.type(self.ntype), self.num_seq[self.seq_idx]).type(torch.int8)
        path = in_0.type(torch.int8) & spike_csg
        # conditional update for seq index when in_0 is 1, which simulates the enable signal.
        self.seq_idx.data = self.seq_idx.add(in_0.type(torch.long))
        
        if self.mode == "unipolar":
            return path.type(self.stype)
        else:
            # generate the conditional spike
            spike_csg = torch.gt(self.in_1_prob.type(self.ntype), self.num_seq[self.seq_idx_inv]).type(torch.int8)
            path_inv = (1 - in_0.type(torch.int8)) & (1 - spike_csg)
            # conditional update for seq_idx_inv
            self.seq_idx_inv.data = self.seq_idx_inv.add(1 - in_0.type(torch.long))
            return (path | path_inv).type(self.stype)
            

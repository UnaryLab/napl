import torch, math

from napl.utils import *
from napl.base import napl_base, hw_params
from napl.module.encoder import gen_num_seq
from loguru import logger



class mul_and(napl_base):
    """
    This module is for unary multiplication with and gate, supporting unipolar/bipolar.
    References:
    1) uGEMM: Unary Computing Architecture for GEMM Applications
    2) uGEMM: Unary Computing for GEMM Applications
    """
    def __init__(
            self,
            config={
                'polarity': 'bipolar',
            }
        ):
        super().__init__(config, ['polarity'], polarity_required=True)
        # combinational AND (unipolar) / XNOR (bipolar): no registers
        self.hw = hw_params(pp_delay=0)


    def reset(self, verbose=False):
        self.timestep_cur = 0


    def forward(self, input_0: torch.tensor, input_1: torch.tensor):
        self.tick()
        # input_0 is a spike tensor
        # input_1 is a spike tensor
        if self.polarity == 'unipolar':
            return (input_0.type(torch.int8) & input_1.type(torch.int8)).type(self.stype)
        else:
            # bipolar multiply = XNOR of the operand spikes
            return input_0.type(torch.int8).bitwise_xor(input_1.type(torch.int8)).bitwise_xor_(1).type(self.stype)


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

        # combinational output; internal seq-index counter(s) are self.width-bit
        # registers that hold no output latency (pp_delay=0). RTL must reset them
        # to 0 to match reset() and instance decorrelated RNGs per polarity path.
        self.hw = hw_params(pp_delay=0)

        # generate the number sequence
        # the sequence is used to compare with the input data
        # this is ntype tensor
        self.num_seq = gen_num_seq(config={'width': self.width,
                                        'generator': self.generator})

        # seq_idx is used later as an enable signal, get update every cycled
        self.seq_idx = torch.nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)
        # Generate two seperate spike generators and two enable signals for bipolar polarity
        if self.polarity == 'bipolar':
            self.seq_idx_inv = torch.nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)

        # only compute the prob of input_1 once in the first call
        self.in_1_prob = None
        self.is_first_call = True


    def reset(self, verbose=False):
        """
        Reset the seq_idx and seq_idx_inv.
        """
        self.timestep_cur = 0
        self.seq_idx.data = torch.zeros(1, dtype=torch.long, device=self.seq_idx.device)
        if self.polarity == 'bipolar':
            self.seq_idx_inv.data = torch.zeros(1, dtype=torch.long, device=self.seq_idx_inv.device)
        self.in_1_prob = None
        self.is_first_call = True


    def forward(self, input_0: torch.tensor, input_1: torch.tensor):
        self.tick()
        # input_0 is a spike tensor
        # input_1 is a binary tensor
        if self.is_first_call is True:
            assert input_1 is not None, logger.error('input_1 is None, please provide a valid input_1 tensor.')
            # cache once as ntype so the per-timestep compare below skips the recast
            self.in_1_prob = ((input_1 + 1) / 2 if self.polarity == 'bipolar' else input_1).type(self.ntype)
            self.is_first_call = False

        # reuse the same int8 view of input_0 across both polarity paths;
        # long.add(int8) promotes to long, so the seq-index counters stay long
        # indices without a separate per-timestep int8->long cast.
        in_0_i8 = input_0.type(torch.int8)
        # generate the conditional spike
        spike_csg = torch.gt(self.in_1_prob, self.num_seq[self.seq_idx]).type(torch.int8)
        path = in_0_i8 & spike_csg
        # conditional update for seq index when input_0 is 1, which simulates the enable signal.
        self.seq_idx.data = self.seq_idx.add(in_0_i8)

        if self.polarity == 'unipolar':
            return path.type(self.stype)
        else:
            # generate the conditional spike
            spike_csg = torch.gt(self.in_1_prob, self.num_seq[self.seq_idx_inv]).type(torch.int8)
            inv_in_0_i8 = 1 - in_0_i8
            path_inv = inv_in_0_i8 & (1 - spike_csg)
            # conditional update for seq_idx_inv
            self.seq_idx_inv.data = self.seq_idx_inv.add(inv_in_0_i8)
            return (path | path_inv).type(self.stype)


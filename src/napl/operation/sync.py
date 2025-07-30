import torch

from napl.base import napl_base
from napl.utils import *


class sync_skewed(napl_base):
    """
    synchronize two input spike trains in a skewed way, please refer to
    1) 'In-Stream Stochastic Division and Square Root via Correlation'
    2) 'In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing'
    """
    def __init__(
            self, 
            config={
                'bitwidth' : 2,
            }
    ):
        super().__init__(config, ['bitwidth'], mode_required=False)

        self.bitwidth=config['bitwidth']
        self.cnt_max = 2**self.bitwidth - 1
        self.cnt = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)
        self.is_first_call = True

        
    def reset(self):
        self.cnt.data = torch.zeros(1, dtype=self.ntype, device=self.cnt.device)
        self.is_first_call = True


    def forward(self, input_1, input_2):
        # input_1 and input_2 are spike tensors
        # this class assume input 1 is smaller than input 2, and input 2 is kept unchanged at output

        # if input 1 and 2  spikes are 01 or 10, sum_in is 1
        input_01_10 = torch.eq(input_1 + input_2, 1).type(self.stype)
        if self.is_first_call:
            # init cnt
            self.cnt.data = torch.zeros_like(input_01_10).type(self.ntype)
            self.is_first_call = False

        cnt_not_min = torch.ne(self.cnt, 0).type(self.stype)
        cnt_not_max = torch.ne(self.cnt, self.cnt_max).type(self.stype)

        # if input is 00/11: input_01_10 == 0
        #   output_1 = input_1
        #   cnt does not change

        # if input is 01/10: input_01_10 == 1
        #   if input_1 is 0: cnt_not_min * (1 - input_1)
        #       if cnt_not_min == 1: cnt has past input_1 saved
        #           output_1 = 1
        #           cnt sub 1
        #       if cnt_not_min == 0: cnt has no past input_1 saved, cnt == 0
        #           output_1 = 0
        #           cnt sub 1 then saturate to 0: no change

        #   if input_1 is 1: (0 - cnt_not_max) * input_1)
        #       if cnt_not_max == 1
        #           output_1 = 0
        #           cnt add 1
        #       if cnt_not_max == 0: cnt == cnt_max
        #           output_1 = 1
        #           cnt add 1 then saturate to cnt_max: no change
        output_1 = input_1.add(input_01_10.mul(cnt_not_min * (1 - input_1) + (0 - cnt_not_max) * input_1))
        self.cnt.data.add_(input_01_10.type(self.ntype).mul(input_1.mul(2).sub(1).type(self.ntype))).clamp_(0, self.cnt_max)
        return output_1, input_2
    

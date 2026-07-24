import torch

from napl.sim.base import napl_base, hw_params


class sync_skewed(napl_base):
    """
    synchronize two input spike streams in a skewed way, please refer to
    1) 'In-Stream Stochastic Division and Square Root via Correlation'
    2) 'In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing'
    """
    def __init__(
            self,
            config={
                'width' : 3,
            }
    ):
        super().__init__(config, ['width'], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        self.width=config['width']
        self.cnt_max = 2**self.width - 1
        self.cnt = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)
        self.is_first_call = True


    def _reset(self):
        self.cnt.data = torch.zeros(1, dtype=self.ntype, device=self.cnt.device)
        self.is_first_call = True


    def forward(self, input_1, input_2):
        # input_1 and input_2 are spike tensors
        # this class assume input 1 is smaller than input 2, and input 2 is kept unchanged at output

        # if input 1 and 2  spikes are 01 or 10, sum_in is 1
        # spikes are {0,1}, so diff is {-1,0,1}: |diff| == (spikes differ) and
        # diff == input_01_10*(2*input_1-1), replacing 6 elementwise kernels with 2
        diff = input_1 - input_2
        input_01_10 = diff.abs()
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
        # select term cnt_not_min*(1-input_1) - cnt_not_max*input_1 rewritten with fewer
        # int8 elementwise ops (input_1 is a {0,1} spike, so this identity is exact)
        select = cnt_not_min - (cnt_not_min + cnt_not_max).mul(input_1)
        output_1 = input_1.add(input_01_10.mul(select))
        # cnt update input_01_10*(2*input_1-1) == diff exactly; add_ into cnt anchors ntype
        self.cnt.data.add_(diff).clamp_(0, self.cnt_max)
        return output_1, input_2

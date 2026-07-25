import torch

from napl.sim.base import napl_base, hw_params


class sync_skewed_int(napl_base):
    """
    synchronize two input spike streams in a skewed way using integer stochastic computing, please refer to
    'VLSI Implementation of Deep Neural Network Using Integral Stochastic Computing'
    """
    def __init__(
            self,
            config={
                'width' : 4,
            }
        ):
        super().__init__(config, ['width'], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        self.width = config['width']
        self.cnt_max = 2**self.width - 1
        self.cnt = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)


    def _reset(self):
        self.cnt.data = torch.zeros(1, dtype=self.ntype, device=self.cnt.device)


    def forward(self, input_1, input_2):
        # input 2 is kept unchanged at output.
        # if input 1 is smaller than input 2, this module works the same as sync_skewed;
        # if input 1 is larger than input 2, spikes of input 1 aggregate in the counter and are
        # released as integer digits (possibly > 1) whenever input 2 spikes, so output 1 is an
        # integer digit stream rather than a {0, 1} spike stream.
        input_2_eq_1 = torch.eq(input_2, 1)
        # accumulate the incoming input 1 spike; type promotion casts input_1 to ntype inside
        # the add, and broadcasts cnt up to the input shape on the first call
        temp_sum = self.cnt + input_1
        # when input 2 spikes, output 1 releases the clipped accumulated count, otherwise 0
        output_1 = input_2_eq_1 * temp_sum.clamp(0, self.cnt_max)
        if temp_sum.shape == output_1.shape:
            # temp_sum is fresh each call: reuse it in place for the counter (all ntype, no promotion)
            self.cnt.data = temp_sum.sub_(output_1).clamp_(0, self.cnt_max)
        else:
            self.cnt.data = (temp_sum - output_1).clamp_(0, self.cnt_max)
        return output_1.type(self.stype), input_2

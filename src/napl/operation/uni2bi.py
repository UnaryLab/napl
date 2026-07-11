import torch

from napl.utils import *
from napl.base import napl_base, hw_params


class uni2bi(napl_base):
    """
    Convert unipolar spike trains to bipolar with scaled addition, please refer to
    "In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing"
    """
    def __init__(
            self,
            config={
                'width' : 3,
            }
    ):
        super().__init__(config, ['width'], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        # width of the accumulator
        self.width = config['width']
        # max value in the accumulator
        self.acc_max = 2**(self.width-1) - 1
        # min value in the accumulator
        self.acc_min = -2**(self.width-1)
        self.accumulator = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)


    def reset(self, verbose=False):
        """
        Reset the accumulator only.
        """
        self.timestep_cur = 0
        self.accumulator.data = torch.zeros(1, dtype=self.ntype, device=self.accumulator.device)


    def forward(self, input):
        self.tick()
        # calculate (input+1)/2
        # input spike trains are [input, 1]
        addend = input.add(1).type(self.ntype)
        acc = self.accumulator.data
        # accumulate in place once the accumulator has broadcast to the input
        # shape; the first call still needs the out-of-place reshape from [1]
        if acc.shape == addend.shape:
            acc.add_(addend).clamp_(self.acc_min, self.acc_max)
        else:
            acc = self.accumulator.data = acc.add(addend).clamp(self.acc_min, self.acc_max)
        # cast the carry-out spike to stype once and reuse it for the in-place
        # acc update (stype->ntype is a safe widening in sub_) and the return,
        # saving one cast versus going through ntype
        output = torch.ge(acc, 2).type(self.stype)
        # acc in [acc_min, acc_max] and output*2 in {0, 2}; subtracting keeps it
        # within range, so the trailing clamp is a no-op and is dropped.
        # alpha=2 folds the *2 into sub_, avoiding the output.mul(2) temporary
        acc.sub_(output, alpha=2)
        return output


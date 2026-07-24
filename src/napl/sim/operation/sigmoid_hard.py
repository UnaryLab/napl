import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import add_any


class sigmoid_hard(napl_base):
    """
    This is a scaled addition (input+1)/2.
    It works for both unipolar and bipolar spike streams.
    """
    def __init__(
        self,
        config={
            'polarity' : 'bipolar'
        },
    ):
        super().__init__(config, ['polarity'], polarity_required=True)
        self.hw = hw_params(pp_delay=0)

        self.scaled_add = add_any({
            'polarity': self.polarity,
            'scale' : 2,
            'width' : 3,
            })


    def forward(self, input: torch.tensor):
        # (input+1)/2: feed the pre-reduced per-timestep sum (input+1) directly
        # (no all-ones stack). For bipolar, fold the +1 into add_any's offset
        # instead: entry=0 -> offset=(0-scale)/2=-1, so acc_delta = input+1,
        # bit-exact with (input+1)-0 from entry=2, minus one full-size alloc
        # and kernel launch per timestep. Unipolar ignores entry (offset stays
        # 0), so it must keep the explicit +1.
        if self.polarity == 'bipolar':
            return self.scaled_add(input, dim=None, entry=0)
        return self.scaled_add(input + 1, dim=None, entry=2)

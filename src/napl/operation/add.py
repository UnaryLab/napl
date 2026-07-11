import torch

from napl.utils import *
from napl.base import napl_base, hw_params
from loguru import logger


class add_any(napl_base):
    """
    This module is any-scale addition.
    Supported polarity: unipolar/bipolar.
    """
    def __init__(
            self,
            config={
                'polarity' : 'bipolar',
                'scale' : 2,
                'width' : 10,
            }
        ):
        super().__init__(config, ['polarity', 'scale', 'width'], polarity_required=True)
        self.hw = hw_params(pp_delay=0)

        # width of the accumulator
        self.width = config['width']
        # max value in the accumulator
        self.acc_max = 2**(self.width-1) - 1
        # min value in the accumulator
        self.acc_min = -2**(self.width-1)

        # the carry scale at the output
        self.scale = torch.nn.Parameter(torch.tensor(config['scale'], dtype=self.ntype), requires_grad=False)
        # accumulation offset
        self.offset = 0
        # accumulator for (PC - offset)
        self.accumulator = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)
        self.is_first_call = True


    def reset(self, verbose=False):
        """
        Reset the accumulator only.
        """
        self.timestep_cur = 0
        self.accumulator.data = torch.zeros(1, dtype=self.ntype, device=self.accumulator.device)
        self.is_first_call = True


    def forward(self, input, entry=None, dim=-1):
        """
        Accumulate one timestep. `input` is the spike tensor to reduce over `dim`;
        pass `dim=None` when `input` is already the per-timestep partial sum (then
        `entry` must be given explicitly for bipolar offset computation).
        """
        self.tick()
        if self.is_first_call:
            if self.polarity == 'bipolar':
                if entry is None:
                    assert dim is not None, \
                        logger.error('add_any with pre-reduced input (dim=None) requires an explicit <entry>.')
                    entry = input.size()[dim]
                self.offset = (entry - self.scale)/2

            self.is_first_call = False

        if dim is None:
            acc_delta = input.type(self.ntype) - self.offset
        else:
            # in-place sub on the freshly-allocated sum (owned temp): same
            # (partial - offset) math as before, one fewer full-size alloc per timestep
            acc_delta = torch.sum(input, dim, dtype=self.ntype)
            acc_delta.sub_(self.offset)
        # in-place add/clamp once the accumulator matches the stream shape (both ntype,
        # so promotion is a no-op); the first timestep must broadcast-expand the (1,)
        # init, which add_ cannot do
        if self.accumulator.shape == acc_delta.shape:
            self.accumulator.add_(acc_delta).clamp_(self.acc_min, self.acc_max)
        else:
            self.accumulator.data = self.accumulator.add(acc_delta).clamp(self.acc_min, self.acc_max)
        output = torch.ge(self.accumulator, self.scale).type(self.ntype)
        # subtract scale only where output==1 (acc>=scale>0), fused, no intermediate
        # alloc; result stays in [0, acc_max] so the post-clamp would be a no-op
        self.accumulator.addcmul_(output, self.scale, value=-1)
        return output.type(self.stype)


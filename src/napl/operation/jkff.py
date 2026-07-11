import torch

from napl.utils import *
from napl.base import napl_base, hw_params


class jkff(napl_base):
    """
    This class is a JK flip-flip.
    """
    def __init__(
            self,
            config={}
        ):
        super().__init__(config, [], polarity_required=False)
        self.hw = hw_params(pp_delay=1)

        self.q = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.q.data = torch.zeros(1, dtype=torch.int8, device=self.q.device)


    def forward(self, input_j: torch.tensor, input_k: torch.tensor):
        self.tick()
        # JK characteristic eq: Q' = (J AND NOT Q) OR (NOT K AND Q). The two
        # terms are mutually exclusive, so for Q in {0,1} this is a plain select:
        # Q' = Q ? (NOT K) : J, a single masked select with no mask or
        # int8-cast temporaries per timestep.
        self.q.data = torch.where(
            self.q.bool(), torch.eq(input_k, 0), torch.ne(input_j, 0)
        ).type(torch.int8)
        return self.q.type(self.stype)


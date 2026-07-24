import torch

from napl.utils import *
from napl.sim.base import napl_base, hw_params


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
        # bool mirror of q; the select consumes this directly so the hot path
        # skips a per-timestep int8->bool cast (q stays int8 for dependents,
        # e.g. sqrt_tracejkff's `(1 - trace) & ...`).
        self.register_buffer('q_b', torch.zeros(1, dtype=torch.bool))


    def _reset(self):
        self.q.data = torch.zeros(1, dtype=torch.int8, device=self.q.device)
        self.q_b = torch.zeros(1, dtype=torch.bool, device=self.q.device)


    def forward(self, input_j: torch.tensor, input_k: torch.tensor):
        # JK characteristic eq: Q' = (J AND NOT Q) OR (NOT K AND Q). The two
        # terms are mutually exclusive, so for Q in {0,1} this is a plain select:
        # Q' = Q ? (NOT K) : J, a single masked select with no mask or
        # int8-cast temporaries per timestep.
        self.q_b = torch.where(self.q_b, torch.eq(input_k, 0), torch.ne(input_j, 0))
        self.q.data = self.q_b.type(torch.int8)
        return self.q.type(self.stype)

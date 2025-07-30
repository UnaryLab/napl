import torch

from napl.utils import *
from napl.base import napl_base


class jkff(napl_base):
    """
    This class is a JK flip-flip.
    """
    def __init__(
            self,
            config={}
        ):
        super().__init__(config, [], mode_required=False)

        self.q = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
    

    def reset(self, verbose=False):
        self.q.data = torch.zeros(1, dtype=torch.int8, device=self.q.device)

    
    def forward(self, input_j: torch.tensor, input_k: torch.tensor):
        j0 = torch.eq(input_j, 0).type(torch.int8)
        j1 = 1 - j0
        k0 = torch.eq(input_k, 0).type(torch.int8)
        k1 = 1 - k0
        
        j0k0 = j0 & k0
        j1k0 = j1 & k0
        j0k1 = j0 & k1
        j1k1 = j1 & k1
        
        self.q.data = j0k0 * self.q + j1k0 * torch.ones_like(input_j, dtype=torch.int8) + j0k1 * torch.zeros_like(input_j, dtype=torch.int8) + j1k1 * (1 - self.q)
        return self.q.type(self.stype)


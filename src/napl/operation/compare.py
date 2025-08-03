import torch

from napl.utils import *
from napl.base import napl_base
from napl.operation import sync_skewed
from loguru import logger


class min_rc(napl_base):
    """
    This class returns the min and argmin using sync_skewed.
    """
    def __init__(
            self, 
            config = {}
    ):
        super().__init__(config, [], polarity_required=False)

        self.dff = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        # default to optimal width
        self.sync = sync_skewed({'width': 2})

    
    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.dff.data = torch.zeros(1, dtype=torch.int8, device=self.dff.device)
        super().reset(verbose)
    

    def forward(self, input_0, input_1):
        self.tick()
        # sync input_0 to input_1
        sync_0, sync_1 = self.sync(input_0, input_1)
        # if sync_0/1 is 01 or 10, enable dff update
        d_enable = sync_0.type(torch.int8) ^ sync_1.type(torch.int8)

        # the next dff value
        # sync_0/1 is 01, meaning input_0 < input_1
        # then 
        # if and_gate == 1, input_1 is larger, and min is 0
        # the and_gate will update dff later
        and_gate = sync_1.type(torch.int8) & d_enable
        
        # generate output
        # if self.dff == 1, input_1 is larger, and min is 0
        output = self.dff * input_0 + (1 - self.dff) * input_1
        
        # update the dff if d_enable is 1
        # this dff value also indicates argmin
        self.dff.data = d_enable * and_gate + (1 - d_enable) * self.dff
        
        # if self.dff == 1, input_1 is larger, and min is 0
        return output.type(self.stype), 1 - self.dff.type(self.stype)


class max_rc(napl_base):
    """
    This class returns the max and argmax using sync_skewed.
    """
    def __init__(
            self, 
            config = {}
    ):
        super().__init__(config, [], polarity_required=False)

        self.dff = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        # default to optimal width
        self.sync = sync_skewed({'width': 2})

    
    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.dff.data = torch.zeros(1, dtype=torch.int8, device=self.dff.device)
        super().reset(verbose)
    

    def forward(self, input_0, input_1):
        self.tick()
        # sync input_0 to input_1
        sync_0, sync_1 = self.sync(input_0, input_1)
        # if sync_0/1 is 01 or 10, enable dff update
        d_enable = sync_0.type(torch.int8) ^ sync_1.type(torch.int8)

        # the next dff value
        # sync_0/1 is 01, meaning input_0 < input_1
        # then 
        # if and_gate == 1, input_1 is larger, and max is 1
        # the and_gate will update dff later
        and_gate = sync_1.type(torch.int8) & d_enable
        
        # generate output
        # if self.dff == 1, input_1 is larger, and max is 1
        output = self.dff * input_1 + (1 - self.dff) * input_0
        
        # update the dff if d_enable is 1
        # this dff value also indicates argmax
        self.dff.data = d_enable * and_gate + (1 - d_enable) * self.dff
        
        # if self.dff == 1, input_1 is larger, and max is 1
        return output.type(self.stype), self.dff.type(self.stype)
    

class lt_rc(napl_base):
    """
    This class returns less than result using sync_skewed.
    """
    def __init__(
            self, 
            config = {}
    ):
        super().__init__(config, [], polarity_required=False)

        self.dff = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        # default to optimal width
        self.sync = sync_skewed({'width': 2})

    
    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.dff.data = torch.zeros(1, dtype=torch.int8, device=self.dff.device)
        super().reset(verbose)
    

    def forward(self, input_0, input_1):
        self.tick()
        # sync input_0 to input_1
        sync_0, sync_1 = self.sync(input_0, input_1)
        # if sync_0/1 is 01 or 10, enable dff update
        d_enable = sync_0.type(torch.int8) ^ sync_1.type(torch.int8)

        # generate output
        # if self.dff == 1, input_0 < input_1
        output = self.dff.clone().detach()

        # update the dff if d_enable is 1
        # if sync_0/1 is 01, input_0 < input_1, update dff to 1
        self.dff.data = d_enable * sync_1.type(torch.int8) + (1 - d_enable) * self.dff
        
        return output.type(self.stype)


class gt_rc(napl_base):
    """
    This class returns greater than result using sync_skewed.
    """
    def __init__(
            self, 
            config = {}
    ):
        super().__init__(config, [], polarity_required=False)

        self.dff = torch.nn.Parameter(torch.ones(1, dtype=torch.int8), requires_grad=False)
        # default to optimal width
        self.sync = sync_skewed({'width': 2})

    
    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.dff.data = torch.ones(1, dtype=torch.int8, device=self.dff.device)
        super().reset(verbose)
    

    def forward(self, input_0, input_1):
        self.tick()
        # sync input_0 to input_1
        sync_0, sync_1 = self.sync(input_0, input_1)
        # if sync_0/1 is 01 or 10, enable dff update
        d_enable = sync_0.type(torch.int8) ^ sync_1.type(torch.int8)

        # generate output
        # if self.dff == 1, input_0 > input_1
        output = self.dff.clone().detach()

        # update the dff if d_enable is 1
        # if sync_0/1 is 10, input_0 > input_1, update dff to 1
        self.dff.data = d_enable * sync_0.type(torch.int8) + (1 - d_enable) * self.dff
        
        return output.type(self.stype)
    

class min_tc(napl_base):
    """
    This class returns the min using AND gate.
    Temporal-coded signals always start with 1s, followed by 0s.
    """
    def __init__(
            self, 
            config = {}
    ):
        super().__init__(config, [], polarity_required=False)

    
    def reset(self, verbose=False):
        self.timestep_cur = 0
    

    def forward(self, input_0, input_1):
        self.tick()
        output = input_0.type(torch.int8) & input_1.type(torch.int8)
        return output.type(self.stype)


class max_tc(napl_base):
    """
    This class returns the max using OR gate.
    Temporal-coded signals always start with 1s, followed by 0s.
    """
    def __init__(
            self, 
            config = {}
    ):
        super().__init__(config, [], polarity_required=False)

    
    def reset(self, verbose=False):
        self.timestep_cur = 0
    

    def forward(self, input_0, input_1):
        self.tick()
        output = input_0.type(torch.int8) | input_1.type(torch.int8)
        return output.type(self.stype)


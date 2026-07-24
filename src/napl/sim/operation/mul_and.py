import torch

from napl.sim.base import napl_base, hw_params


class mul_and(napl_base):
    """
    This module is for unary multiplication with and gate, supporting unipolar/bipolar.
    References:
    1) uGEMM: Unary Computing Architecture for GEMM Applications
    2) uGEMM: Unary Computing for GEMM Applications
    """
    def __init__(
            self,
            config={
                'polarity': 'bipolar',
            }
        ):
        super().__init__(config, ['polarity'], polarity_required=True)
        # combinational AND (unipolar) / XNOR (bipolar): no registers
        self.hw = hw_params(pp_delay=0)


    def forward(self, input_0: torch.tensor, input_1: torch.tensor):
        # input_0 is a spike tensor
        # input_1 is a spike tensor
        if self.polarity == 'unipolar':
            return (input_0.type(torch.int8) & input_1.type(torch.int8)).type(self.stype)
        else:
            # bipolar multiply = XNOR of the operand spikes
            return input_0.type(torch.int8).bitwise_xor(input_1.type(torch.int8)).bitwise_xor_(1).type(self.stype)

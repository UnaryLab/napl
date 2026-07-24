import torch

from napl.sim.base import napl_base, hw_params


class mul_gaines(napl_base):
    """
    This module is for Gaines stochastic multiplication, supporting unipolar/bipolar.
    Gate-identical to mul_and (unipolar AND / bipolar XNOR); kept as a standalone
    module to mirror UnarySim's GainesMul.
    References:
    1) B. R. Gaines, "Stochastic Computing Systems"
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

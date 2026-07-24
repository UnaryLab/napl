from napl.sim.base import napl_base, hw_params
from napl.sim.operation import add_any


class relu_sat(napl_base):
    """
    ReLU activation by saturating the spike value to 0
    The spike stream should always bipolar and rate coded
    """
    def __init__(
            self,
            config={}
    ):
        super().__init__(config, [], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        # default to optimal width
        self.sub_1 = add_any({'polarity': 'bipolar', 'scale': 1, 'width': 3})
        self.add_1 = add_any({'polarity': 'bipolar', 'scale': 1, 'width': 3})


    def forward(self, input):
        # sub_1 moves input from [-1, 1] to [-1, 0]: the 2-row reduction sum([input, 0])
        # is just `input`, so feed the pre-reduced partial sum (dim=None) with the row
        # count as <entry> and skip the per-timestep stack alloc + sum reduction.
        sub_1_out = self.sub_1(input, entry=2, dim=None)
        # add_1 moves input from [-1, 0] to [0, 1]: sum([sub_1_out, 1]) == sub_1_out + 1
        output = self.add_1(sub_1_out + 1, entry=2, dim=None)
        return output

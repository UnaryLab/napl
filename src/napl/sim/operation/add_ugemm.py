import torch

from napl.sim.base import napl_base, hw_params


class add_ugemm(napl_base):
    """
    This module is uGEMM-style addition (scaled or non-scaled).
    Scaled mode emits a carry whenever the running spike count reaches the
    entry count along the reduced dim (output ~= mean of inputs); non-scaled
    mode compares the offset-corrected count against the emitted-spike count
    (output ~= clipped sum of inputs).
    Supported polarity: unipolar/bipolar.
    """
    def __init__(
            self,
            config={
                'polarity' : 'bipolar',
                'scaled' : True,
            }
        ):
        super().__init__(config, ['polarity', 'scaled'], polarity_required=True)
        self.hw = hw_params(pp_delay=0)

        # whether the addition is scaled (carry-out per acc_bound spikes)
        self.scaled = config['scaled']
        # upper bound of the accumulation counter (= entry count along dim)
        self.acc_bound = 0
        # per-timestep accumulation offset (non-scaled bipolar only)
        self.offset = 0
        # accumulator of the per-timestep partial counts
        self.accumulator = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)
        # count of already-emitted output spikes (non-scaled mode)
        if not self.scaled:
            self.out_accumulator = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)
        self.is_first_call = True


    def _reset(self):
        """
        Reset the accumulators only.
        """
        self.accumulator.data = torch.zeros(1, dtype=self.ntype, device=self.accumulator.device)
        if not self.scaled:
            self.out_accumulator.data = torch.zeros(1, dtype=self.ntype, device=self.out_accumulator.device)
        self.is_first_call = True


    def forward(self, input, dim=-1):
        """
        Accumulate one timestep. `input` is the spike tensor to reduce over `dim`.
        """
        if self.is_first_call:
            self.acc_bound = input.size()[dim]
            if self.polarity == 'bipolar':
                self.offset = (self.acc_bound - 1) / 2
            self.is_first_call = False

        acc_delta = torch.sum(input, dim, dtype=self.ntype)
        # out-of-place add broadcasts the (1,) init up to the stream shape on the
        # first timestep (napl accumulator idiom); in-place afterwards (same ntype)
        if self.accumulator.shape == acc_delta.shape:
            self.accumulator.add_(acc_delta)
        else:
            self.accumulator.data = self.accumulator.add(acc_delta)

        # compare -> stype directly (one cast); sub_/add_ promote the int8 spike
        # to the float32 destination, so results are unchanged
        if self.scaled:
            output = torch.ge(self.accumulator, self.acc_bound).type(self.stype)
            self.accumulator.sub_(output, alpha=self.acc_bound)
        else:
            self.accumulator.sub_(self.offset)
            output = torch.gt(self.accumulator, self.out_accumulator).type(self.stype)
            if self.out_accumulator.shape == output.shape:
                self.out_accumulator.add_(output)
            else:
                self.out_accumulator.data = self.out_accumulator.add(output)

        return output

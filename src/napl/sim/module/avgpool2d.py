import torch

from napl.sim.base import napl_base


class avgpool2d(napl_base):
    """
    Streaming unary 2d average pooling based on scaled addition.
    Each timestep pools one spike tensor with AvgPool2d (window mean in [0, 1]),
    accumulates it, and emits an output spike wherever the accumulator reaches 1
    (then subtracts the emitted spike), so the output stream rate equals the
    window-mean input rate. The mapping rate -> value is affine for both
    polarities, so the same math serves unipolar and bipolar streams.
    UnarySim: FSUAvgPool2d.
    """
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None,
                 config={'polarity': 'bipolar'}):
        super().__init__(config, ['polarity'], polarity_required=True)
        self.avgpool2d = torch.nn.AvgPool2d(kernel_size, stride=stride, padding=padding,
                                            ceil_mode=ceil_mode,
                                            count_include_pad=count_include_pad,
                                            divisor_override=divisor_override)
        # scalar accumulator; broadcasts up to the pooled output shape on the
        # first forward() (napl broadcast idiom, no pre-sized input_shape needed)
        self.accumulator = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)

    def _reset(self):
        self.accumulator.data = torch.zeros(1, dtype=self.ntype, device=self.accumulator.device)

    def forward(self, input_spike):
        # input_spike: (batch, channel, H, W) spike tensor for the current timestep
        delta = self.avgpool2d(input_spike.type(self.ntype))
        if self.accumulator.shape == delta.shape:
            self.accumulator.add_(delta)
        else:
            # first timestep: broadcast-expand the (1,) init out of place
            self.accumulator.data = self.accumulator.add(delta)
        # single cast to stype: sub_ promotes the 0/1 spike into the ntype
        # accumulator in place, saving one full-tensor cast per timestep
        output = torch.ge(self.accumulator, 1).type(self.stype)
        self.accumulator.sub_(output)
        return output

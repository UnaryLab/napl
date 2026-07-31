import torch

from napl.utils import *
from napl.sim.base import napl_base, hw_params


class bi2uni(napl_base):
    """
    Convert a bipolar rate-coded spike stream to unipolar form.

    Use this stateful converter when a bipolar stream must feed a unipolar
    operation for a nonnegative represented value. It applies the inverse
    one-density mapping ``2p - 1`` through a bounded accumulator.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import bi2uni

        converter = bi2uni({'width': 2})
        output = converter(torch.tensor([1, 0], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*.
    """
    def __init__(
            self,
            config={
                'width' : 2,
            }
    ):
        """
        Configure the bounded conversion accumulator.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Signed accumulator width in bits; the default is ``2``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['width'], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        # width of the accumulator
        self.width = config['width']
        # max value in the accumulator
        self.acc_max = 2**(self.width-1) - 1
        # min value in the accumulator
        self.acc_min = -2**(self.width-1)
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))


    def _reset(self):
        """
        Clear the local conversion accumulator.
        """
        self.accumulator.resize_(1).zero_()


    def forward(self, input):
        """
        Convert one timestep of bipolar input spikes.

        Args:
            input: Tensor of current 0/1 bipolar-encoded spikes.

        Returns:
            A tensor of unipolar-encoded output spikes with the same shape. The
            call updates the bounded accumulator.

        **Example:**

        .. code-block:: python

            output = converter(torch.tensor([1, 0], dtype=torch.int8))
        """
        # calculate (2*input-1)/1
        # input spike streams are [input, input, 0]
        # fuse acc + (2*input - 1), then clamp. ntype destination promotes the int8 input
        # (alpha=2), so the explicit .type(ntype) cast is unnecessary.
        acc = self.accumulator
        if acc.shape == input.shape:
            # steady state: update the accumulator buffer in place (no per-timestep alloc).
            acc = acc.add_(input, alpha=2).sub_(1).clamp_(self.acc_min, self.acc_max)
        else:
            # first call: broadcast the (1,) accumulator up to the input shape out of place.
            acc = acc.add(input, alpha=2).sub_(1).clamp_(self.acc_min, self.acc_max)
        # output as stype directly; acc.sub_ promotes int8 up to float32 destination (no truncation),
        # which drops the separate stype cast at return.
        output = torch.ge(acc, 1).type(self.stype)
        # acc is already in [acc_min, acc_max] and output in {0,1} with output==1 only when acc>=1,
        # so acc-output stays in range; the trailing clamp is a provable no-op.
        acc.sub_(output)
        if self.accumulator.shape != acc.shape:
            self.accumulator.resize_as_(acc).copy_(acc.detach())
        return output

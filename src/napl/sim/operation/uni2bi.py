import torch

from napl.utils import *
from napl.sim.base import napl_base, hw_params


class uni2bi(napl_base):
    """
    Convert a unipolar rate-coded spike stream to bipolar form.

    Use this stateful converter when a unipolar stream must feed a bipolar
    operation. It preserves the represented value by producing a bipolar stream
    whose one-density is the unipolar input value mapped by ``(x + 1) / 2``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import uni2bi

        converter = uni2bi({'width': 3})
        output = converter(torch.tensor([1, 0], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*.
    """
    def __init__(
            self,
            config={
                'width' : 3,
            }
    ):
        """
        Configure the bounded conversion accumulator.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Signed accumulator width in bits; the default is ``3``.
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
        Convert one timestep of unipolar input spikes.

        Args:
            input: Tensor of current 0/1 unipolar spikes.

        Returns:
            A tensor of bipolar-encoded output spikes with the same shape. The
            call updates the bounded accumulator.

        **Example:**

        .. code-block:: python

            output = converter(torch.tensor([1, 0], dtype=torch.int8))
        """
        # calculate (input+1)/2
        # input spike streams are [input, 1]; the +1 is folded into a second
        # in-place add so no addend/cast temporaries are allocated (in-place
        # stype->ntype add is a safe widening into the ntype accumulator)
        acc = self.accumulator
        # accumulate in place once the accumulator has broadcast to the input
        # shape; the first call still needs the out-of-place reshape from [1]
        if acc.shape == input.shape:
            acc.add_(input).add_(1).clamp_(self.acc_min, self.acc_max)
        else:
            acc = acc.add(input).add_(1).clamp_(self.acc_min, self.acc_max)
        # cast the carry-out spike to stype once and reuse it for the in-place
        # acc update (stype->ntype is a safe widening in sub_) and the return,
        # saving one cast versus going through ntype
        output = torch.ge(acc, 2).type(self.stype)
        # acc in [acc_min, acc_max] and output*2 in {0, 2}; subtracting keeps it
        # within range, so the trailing clamp is a no-op and is dropped.
        # alpha=2 folds the *2 into sub_, avoiding the output.mul(2) temporary
        acc.sub_(output, alpha=2)
        if self.accumulator.shape != acc.shape:
            self.accumulator.resize_as_(acc).copy_(acc.detach())
        return output

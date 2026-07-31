import torch

from napl.sim.base import napl_base, hw_params


class sync_skewed_int(napl_base):
    """
    Synchronize streams with an integral stochastic output.

    Use this stateful variant when the first stream may exceed the second. It
    accumulates first-stream spikes and releases a bounded integer digit whenever
    the second stream spikes, while passing the second stream through unchanged.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import sync_skewed_int

        sync = sync_skewed_int({'width': 4})
        first, second = sync(torch.tensor([1], dtype=torch.int8),
                             torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *VLSI Implementation of Deep Neural Network Using Integral Stochastic Computing*.
    """
    def __init__(
            self,
            config={
                'width' : 4,
            }
        ):
        """
        Configure the integral accumulation counter.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Counter width in bits, giving a maximum output digit of ``2**width - 1``; the default is ``4``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['width'], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        self.width = config['width']
        self.cnt_max = 2**self.width - 1
        self.register_buffer('cnt', torch.zeros(1, dtype=self.ntype))


    def _reset(self):
        """
        Clear the local accumulated first-stream count.
        """
        self.cnt.resize_(1).zero_()


    def forward(self, input_1, input_2):
        """
        Accumulate and conditionally release one integer-coded timestep.

        Args:
            input_1: Current first-stream spike or integer digit tensor.
            input_2: Current 0/1 release-control spike tensor.

        Returns:
            A pair ``(output_1, input_2)``. ``output_1`` contains the released
            bounded integer digits, and the second tensor is unchanged. Unreleased
            counts remain in the local counter.

        **Example:**

        .. code-block:: python

            first, second = sync(torch.tensor([1], dtype=torch.int8),
                                 torch.tensor([1], dtype=torch.int8))
        """
        # input 2 is kept unchanged at output.
        # if input 1 is smaller than input 2, this module works the same as sync_skewed;
        # if input 1 is larger than input 2, spikes of input 1 aggregate in the counter and are
        # released as integer digits (possibly > 1) whenever input 2 spikes, so output 1 is an
        # integer digit stream rather than a {0, 1} spike stream.
        input_2_eq_1 = torch.eq(input_2, 1)
        # accumulate the incoming input 1 spike; type promotion casts input_1 to ntype inside
        # the add, and broadcasts cnt up to the input shape on the first call
        temp_sum = self.cnt + input_1
        # when input 2 spikes, output 1 releases the clipped accumulated count, otherwise 0
        output_1 = input_2_eq_1 * temp_sum.clamp(0, self.cnt_max)
        if temp_sum.shape == output_1.shape:
            # temp_sum is fresh each call: reuse it in place for the counter (all ntype, no promotion)
            updated = temp_sum.sub_(output_1).clamp_(0, self.cnt_max)
        else:
            updated = (temp_sum - output_1).clamp_(0, self.cnt_max)
        if self.cnt.shape == updated.shape:
            self.cnt.copy_(updated.detach())
        else:
            self.cnt.resize_as_(updated).copy_(updated.detach())
        return output_1.type(self.stype), input_2

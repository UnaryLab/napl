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
        #: Hardware latency and timing metadata for the combinational synchronizer.
        self.hw = hw_params(pp_delay=0)

        #: Width of the stored integer stream-skew counter in bits.
        self.width = config['width']
        #: Largest accumulated first-stream count retained by the synchronizer.
        self.cnt_max = 2**self.width - 1
        #: Per-element first-stream count awaiting release by the second stream.
        self.cnt: torch.Tensor
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
        # input_2 passes through unchanged. Excess input_1 spikes accumulate and are released
        # as integer digits when input_2 spikes.
        input_2_eq_1 = torch.eq(input_2, 1)
        # ntype addition broadcasts the scalar counter on first use.
        temp_sum = self.cnt + input_1
        output_1 = input_2_eq_1 * temp_sum.clamp(0, self.cnt_max)
        if temp_sum.shape == output_1.shape:
            updated = temp_sum.sub_(output_1).clamp_(0, self.cnt_max)
        else:
            updated = (temp_sum - output_1).clamp_(0, self.cnt_max)
        if self.cnt.shape == updated.shape:
            self.cnt.copy_(updated.detach())
        else:
            self.cnt.resize_as_(updated).copy_(updated.detach())
        return output_1.type(self.stype), input_2

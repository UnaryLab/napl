import torch

from napl.sim.base import napl_base, hw_params


class sync_skewed(napl_base):
    """
    Correlate two unipolar streams with skewed synchronization.

    Use this stateful synchronizer before correlation-based operations when the
    first stream's represented value does not exceed the second's. It retimes
    the first stream while passing the second stream through unchanged.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import sync_skewed

        sync = sync_skewed({'width': 3})
        first, second = sync(torch.tensor([0], dtype=torch.int8),
                             torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Stochastic Division and Square Root via Correlation*.

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*.
    """
    def __init__(
            self,
            config={
                'width' : 3,
            }
    ):
        """
        Configure the skew buffer counter.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Counter width in bits, giving a maximum stored skew of ``2**width - 1``; the default is ``3``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['width'], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        self.width=config['width']
        self.cnt_max = 2**self.width - 1
        self.register_buffer('cnt', torch.zeros(1, dtype=self.ntype))
        self.is_first_call = True


    def _reset(self):
        """
        Clear the local skew counter and first-call shape state.
        """
        self.cnt.resize_(1).zero_()
        self.is_first_call = True


    def forward(self, input_1, input_2):
        """
        Synchronize one timestep of two input streams.

        Args:
            input_1: Current 0/1 spikes from the stream with the smaller or equal
                represented value.
            input_2: Current 0/1 spikes from the reference stream.

        Returns:
            A pair ``(output_1, input_2)``. ``output_1`` is the skew-adjusted
            first stream and the second tensor is returned unchanged. The call
            updates the saturating skew counter.

        **Example:**

        .. code-block:: python

            first, second = sync(torch.tensor([0], dtype=torch.int8),
                                 torch.tensor([1], dtype=torch.int8))
        """
        # input_1 and input_2 are spike tensors
        # this class assume input 1 is smaller than input 2, and input 2 is kept unchanged at output

        # if input 1 and 2  spikes are 01 or 10, sum_in is 1
        # spikes are {0,1}, so diff is {-1,0,1}: |diff| == (spikes differ) and
        # diff == input_01_10*(2*input_1-1), replacing 6 elementwise kernels with 2
        diff = input_1 - input_2
        input_01_10 = diff.abs()
        if self.is_first_call:
            # init cnt
            self.cnt.resize_as_(input_01_10).zero_()
            self.is_first_call = False

        cnt_not_min = torch.ne(self.cnt, 0).type(self.stype)
        cnt_not_max = torch.ne(self.cnt, self.cnt_max).type(self.stype)

        # if input is 00/11: input_01_10 == 0
        #   output_1 = input_1
        #   cnt does not change

        # if input is 01/10: input_01_10 == 1
        #   if input_1 is 0: cnt_not_min * (1 - input_1)
        #       if cnt_not_min == 1: cnt has past input_1 saved
        #           output_1 = 1
        #           cnt sub 1
        #       if cnt_not_min == 0: cnt has no past input_1 saved, cnt == 0
        #           output_1 = 0
        #           cnt sub 1 then saturate to 0: no change

        #   if input_1 is 1: (0 - cnt_not_max) * input_1)
        #       if cnt_not_max == 1
        #           output_1 = 0
        #           cnt add 1
        #       if cnt_not_max == 0: cnt == cnt_max
        #           output_1 = 1
        #           cnt add 1 then saturate to cnt_max: no change
        # select term cnt_not_min*(1-input_1) - cnt_not_max*input_1 rewritten with fewer
        # int8 elementwise ops (input_1 is a {0,1} spike, so this identity is exact)
        select = cnt_not_min - (cnt_not_min + cnt_not_max).mul(input_1)
        output_1 = input_1.add(input_01_10.mul(select))
        # cnt update input_01_10*(2*input_1-1) == diff exactly; add_ into cnt anchors ntype
        self.cnt.add_(diff).clamp_(0, self.cnt_max)
        return output_1, input_2

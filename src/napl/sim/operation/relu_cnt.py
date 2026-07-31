import torch

from napl.sim.base import napl_base, hw_params


class relu_cnt(napl_base):
    """
    Apply ReLU to a bipolar rate-coded stream with a saturating counter.

    Use this streaming kernel when a bounded counter estimate of the decoded
    input sign is suitable. The output remains a bipolar 0/1 spike stream.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import relu_cnt

        operation = relu_cnt()
        output = operation(torch.tensor([0.0, 1.0]))
    """
    def __init__(
            self,
            config={
                'width' : 3,
            }
    ):
        """
        Configure the ReLU state counter.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Counter bit width; the default is ``3``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['width'], polarity_required=False)
        self.hw = hw_params(pp_delay=0)

        self.width = config['width']

        self.buf_max = 2**self.width - 1
        self.buf_half = 2**(self.width - 1)
        self.register_buffer('acc', torch.zeros(1, dtype=self.ntype).fill_(2**(self.width - 1)))


    def _reset(self):
        """
        Restore the accumulator to its half-scale initial state.
        """
        self.acc.resize_(1).fill_(2**(self.width - 1))


    def forward(self, input):
        """
        Process one timestep of a bipolar rate-coded stream.

        The call produces the current ReLU spike, then updates and saturates
        the accumulator from that output spike.

        Args:
            input: Tensor of current 0/1 input spikes.

        Returns:
            Bipolar 0/1 output spike tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        # below_half is the complement of (acc >= half); lt avoids the extra (1 - ge) step.
        below_half = torch.lt(self.acc, self.buf_half)
        # only when input is 0 and flag is 1, output 0; otherwise 1
        # int8 | bool yields int8, so below_half needs no separate cast
        output = input.type(torch.int8) | below_half
        # update the accumulator based on output, thus acc update is after output generation
        # acc += 2*output - 1, then clamp; fused/in-place to avoid per-timestep intermediate allocations
        if self.acc.shape == output.shape:
            self.acc.add_(output, alpha=2).sub_(1).clamp_(0, self.buf_max)
        else:
            updated = self.acc.add(output, alpha=2).sub_(1).clamp_(0, self.buf_max)
            self.acc.resize_as_(updated).copy_(updated.detach())
        return output.type(self.stype)

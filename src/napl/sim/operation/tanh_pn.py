import torch

from napl.sim.base import napl_base, hw_params


class tanh_pn(napl_base):
    """
    Approximate ``tanh(N * x / 2)`` with an ``N``-state FSM.

    Here ``N = 2**depth``. Use this streaming kernel with bipolar rate-coded
    input when a saturating-counter tanh implementation is desired.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import tanh_pn

        operation = tanh_pn({'depth': 5})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        B. D. Brown and H. C. Card, *Stochastic neural computation I: Computational elements*.
    """
    def __init__(
            self,
            config={
                'depth' : 5,
            }
    ):
        """
        Configure the FSM state count.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **depth**: Counter bit width, giving ``2**depth`` states; the default is ``5``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['depth'], polarity_required=False)
        self.hw = hw_params(pp_delay=1)

        self.depth = config['depth']

        self.cnt_max = 2**self.depth - 1
        self.cnt_half = 2**(self.depth - 1)
        # scalar state; broadcasts up to the input shape on the first forward()
        self.register_buffer('cnt', torch.zeros(1, dtype=self.ntype).fill_(self.cnt_half))


    def _reset(self):
        """
        Restore the FSM counter to its half-scale initial state.
        """
        self.cnt.resize_(1).fill_(self.cnt_half)


    def forward(self, input):
        """
        Process one timestep of a bipolar rate-coded stream.

        The output is derived from the pre-update counter state. The call then
        increments on a 1-spike or decrements on a 0-spike and saturates the FSM
        counter at its limits.

        Args:
            input: Tensor of current 0/1 bipolar input spikes.

        Returns:
            Bipolar 0/1 tanh output spike tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        # output looks at the state before this timestep's update
        output = torch.ge(self.cnt, self.cnt_half).type(self.stype).expand_as(input)
        # cnt += 2*input - 1, then clamp; in-place once shape matches (first call
        # broadcasts (1,) -> input shape out-of-place)
        if self.cnt.shape == input.shape:
            self.cnt.add_(input, alpha=2).sub_(1).clamp_(0, self.cnt_max)
        else:
            updated = self.cnt.add(input.type(self.ntype), alpha=2).sub_(1).clamp_(0, self.cnt_max)
            self.cnt.resize_as_(updated).copy_(updated.detach())
        return output

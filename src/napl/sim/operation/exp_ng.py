import torch

from napl.sim.base import napl_base, hw_params


class exp_ng(napl_base):
    """
    Approximate ``exp(-2 * gain * x)`` with a saturating-counter FSM.

    Use this streaming kernel with a bipolar rate-coded input representing a
    non-negative value. It produces a unipolar rate-coded exponential.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import exp_ng

        operation = exp_ng()
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        B. D. Brown and H. C. Card, *Stochastic neural computation I: Computational elements*.
    """
    def __init__(
            self,
            config={
                'depth': 5,
                'gain': 1,
            }
    ):
        """
        Configure the exponentiation FSM.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **depth**: Counter bit width; the default is ``5``.
              - **gain**: Exponential gain and number of upper counter states that emit zero; the default is ``1``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['depth'], polarity_required=False)
        # output is combinational from the state counter; input reaches it one cycle later
        self.hw = hw_params(pp_delay=1)

        self.depth = config['depth']
        self.gain = config.get('gain', 1)

        self.cnt_max = 2**self.depth - 1
        # emit a 1-spike while the counter is below this threshold (top `gain` states emit 0)
        self.thd = 2**self.depth - self.gain
        self.register_buffer('cnt', torch.zeros(1, dtype=self.ntype).fill_(2**(self.depth - 1)))


    def _reset(self):
        """
        Restore the saturating counter to its half-scale initial state.
        """
        self.cnt.resize_(1).fill_(2**(self.depth - 1))


    def forward(self, input):
        """
        Process one timestep of a bipolar rate-coded input stream.

        The output is derived from the pre-update counter state. The call then
        increments the counter on a 1-spike and decrements it on a 0-spike,
        with saturation at the configured counter limits.

        Args:
            input: Tensor of current 0/1 bipolar input spikes.

        Returns:
            Unipolar 0/1 output spike tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        # output reflects the state before this timestep's input is absorbed
        output = torch.lt(self.cnt, self.thd).type(self.stype)
        if output.shape != input.shape:
            output = torch.zeros_like(input) + output
        # count up on a 1-spike, down on a 0-spike, saturating at [0, 2**depth - 1]
        if self.cnt.shape == input.shape:
            # steady state: in-place on the ntype counter (promotion is a no-op)
            self.cnt.add_(input, alpha=2).sub_(1).clamp_(0, self.cnt_max)
        else:
            # first call: out-of-place add broadcasts the scalar counter to input shape
            updated = self.cnt.add(input.type(self.ntype), alpha=2).sub_(1).clamp_(0, self.cnt_max)
            self.cnt.resize_as_(updated).copy_(updated.detach())
        return output

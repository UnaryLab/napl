import torch

from napl.sim.base import napl_base


class exp_n2g(napl_base):
    r"""
    Compute a bipolar-input exponential with a saturating counter.

    The target rate-domain operation is

    .. math::

       y = \exp(-2Gx),

    with G = ``gain``. A saturating state counter of ``2**depth`` states
    approximates that target, and the approximation error shrinks as
    **depth** grows.

    The input is bipolar 0/1 rate-coded and the output is a unipolar 0/1
    rate-coded stream.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import exp_n2g

        operation = exp_n2g()
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *Stochastic Neural Computation I: Computational Elements*, IEEE Transactions on Computers, 2001.
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
        super().__init__(config, ['depth'], optional_key_list=['polarity', 'gain'], polarity_required=False)

        #: Saturating state-counter width in bits.
        self.depth = config['depth']
        #: Exponential gain and number of upper counter states that emit zero.
        self.gain = config.get('gain', 1)

        #: Largest value retained by the state counter.
        self.cnt_max = 2**self.depth - 1
        # The top gain counter states emit 0; all lower states emit 1.
        #: Exclusive counter threshold below which the output spike is one.
        self.thd = 2**self.depth - self.gain
        #: Saturating state counter updated by each bipolar input spike.
        self.cnt: torch.Tensor
        self.register_buffer('cnt', torch.zeros(1, dtype=self.ntype).fill_(2**(self.depth - 1)))
        # Output uses the counter state before the current update, giving one-cycle latency.
        #: Hardware latency and timing metadata for the registered output.
        self.hw.pp_delay = 1

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'bipolar', 'output': 'unipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


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
        output = torch.lt(self.cnt, self.thd).type(self.stype)
        if output.shape != input.shape:
            output = torch.zeros_like(input) + output
        # The scalar initial counter broadcasts out of place; matching shapes update in place.
        if self.cnt.shape == input.shape:
            self.cnt.add_(input, alpha=2).sub_(1).clamp_(0, self.cnt_max)
        else:
            updated = self.cnt.add(input.type(self.ntype), alpha=2).sub_(1).clamp_(0, self.cnt_max)
            self.cnt.resize_as_(updated).copy_(updated.detach())
        return output

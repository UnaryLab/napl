import torch

from napl.sim.base import napl_base, hw_params


class tanh_pn(napl_base):
    r"""
    Compute a bipolar tanh FSM with a saturating counter.

    The target rate-domain operation is

    .. math::

       f(x) = \tanh\left(\frac{N x}{2}\right),\qquad N=2^{depth}.

    A saturating counter with :math:`N` states approximates it, so the accuracy
    improves with **depth** and the output lags a changing input while the
    counter settles. The input and output are bipolar 0/1 rate-coded streams.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import tanh_pn

        operation = tanh_pn({'depth': 5})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *Stochastic Neural Computation I: Computational Elements*, IEEE Transactions on Computers, 2001.
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
        super().__init__(config, ['depth'], optional_key_list=['polarity'], polarity_required=False)

        #: Width of the saturating tanh state counter in bits.
        self.depth = config['depth']

        #: Largest value retained by the tanh state counter.
        self.cnt_max = 2**self.depth - 1
        #: Half-scale counter value restored by :meth:`_reset`.
        self.cnt_half = 2**(self.depth - 1)
        # The scalar initial counter broadcasts to the input shape on first use.
        #: Saturating state counter that drives the bipolar tanh output.
        self.cnt: torch.Tensor
        self.register_buffer('cnt', torch.zeros(1, dtype=self.ntype).fill_(self.cnt_half))
        #: Hardware latency and timing metadata for the registered tanh output.
        self.hw = hw_params(pp_delay=1)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'bipolar', 'output': 'bipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


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
        # Output reflects the pre-update counter state.
        output = torch.ge(self.cnt, self.cnt_half).type(self.stype).expand_as(input)
        if self.cnt.shape == input.shape:
            self.cnt.add_(input, alpha=2).sub_(1).clamp_(0, self.cnt_max)
        else:
            updated = self.cnt.add(input.type(self.ntype), alpha=2).sub_(1).clamp_(0, self.cnt_max)
            self.cnt.resize_as_(updated).copy_(updated.detach())
        return output

import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import add_any


class sigmoid_hard(napl_base):
    r"""
    Apply the hard-sigmoid transform to a unipolar or bipolar spike stream.

    The precise target rate-domain transform is

    .. math::

       f(x) = \frac{x+1}{2}.

    Let z_t = x_t + 1 be the mode-specific input to the scale-two adder.
    With a_0 = 0, its exact carry recurrence is

    .. math::

       \begin{aligned}
       \tilde a_t &= \operatorname{clip}(a_{t-1}+z_t,-8,7),\\
       y_t &= \mathbf{1}\{\tilde a_t\geq 2\},\qquad
       a_t = \tilde a_t-2y_t.
       \end{aligned}

    Thus the rate-domain operation is E[y] = (E[x]+1)/2.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import sigmoid_hard

        operation = sigmoid_hard({'polarity': 'bipolar'})
        output = operation(torch.tensor([0.0, 1.0]))
    """


    def __init__(
        self,
        config={
            'polarity' : 'bipolar'
        },
    ):
        """
        Configure the spike-stream encoding.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['polarity'], polarity_required=True)

        #: Scaled unary adder that implements the affine sigmoid transform.
        self.scaled_add = add_any({
            'polarity': self.polarity,
            'scale' : 2,
            'width' : 4,
            })
        #: Hardware latency and timing metadata for the composed hard sigmoid.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Reset no class-owned state; :meth:`reset` resets the child scaled adder.
        """
        pass


    def forward(self, input: torch.tensor):
        """
        Process one timestep of a unipolar or bipolar input stream.

        The call advances the internal scaled adder and returns the spike for
        the transformed value ``(x + 1) / 2``.

        Args:
            input: Tensor of current 0/1 input spikes.

        Returns:
            0/1 output spike tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        # Bipolar entry=0 folds the +1 term of (input + 1) / 2 into the offset.
        # Unipolar mode requires the explicit +1 because its offset is zero.
        if self.polarity == 'bipolar':
            return self.scaled_add(input, dim=None, entry=0)
        return self.scaled_add(input + 1, dim=None, entry=2)

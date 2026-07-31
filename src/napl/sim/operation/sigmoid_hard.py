import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import add_any


class sigmoid_hard(napl_base):
    """
    Apply the hard-sigmoid transform ``(x + 1) / 2`` to a spike stream.

    Use this streaming scaled-adder kernel with either unipolar or bipolar
    rate-coded input when a linear hard-sigmoid approximation is sufficient.

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
        self.hw = hw_params(pp_delay=0)

        self.scaled_add = add_any({
            'polarity': self.polarity,
            'scale' : 2,
            'width' : 3,
            })


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
        # (input+1)/2: feed the pre-reduced per-timestep sum (input+1) directly
        # (no all-ones stack). For bipolar, fold the +1 into add_any's offset
        # instead: entry=0 -> offset=(0-scale)/2=-1, so acc_delta = input+1,
        # bit-exact with (input+1)-0 from entry=2, minus one full-size alloc
        # and kernel launch per timestep. Unipolar ignores entry (offset stays
        # 0), so it must keep the explicit +1.
        if self.polarity == 'bipolar':
            return self.scaled_add(input, dim=None, entry=0)
        return self.scaled_add(input + 1, dim=None, entry=2)

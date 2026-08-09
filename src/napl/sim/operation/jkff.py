import torch

from napl.sim.base import napl_base


class jkff(napl_base):
    r"""
    Apply the JK flip-flop state transition to tensor inputs.

    Each tensor position follows the JK truth table

    .. math::

       \begin{array}{cc|cl}
       j_t & k_t & q_t & \\
       \hline
       0 & 0 & q_{t-1} & \text{(hold)} \\
       0 & 1 & 0 & \text{(reset)} \\
       1 & 0 & 1 & \text{(set)} \\
       1 & 1 & \overline{q_{t-1}} & \text{(toggle)}
       \end{array}

    The updated q_t is returned at each timestep.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import jkff

        flip_flop = jkff()
        q = flip_flop(torch.tensor([1], dtype=torch.int8),
                      torch.tensor([0], dtype=torch.int8))
    """


    def __init__(
            self,
            config={}
        ):
        """
        Construct a zero-initialized JK flip-flop.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping with no operation-specific keys.
              **name** may optionally label the instance; the default is ``{}``.
        """
        super().__init__(config, [], optional_key_list=['polarity'], polarity_required=False)

        #: Current JK flip-flop output state.
        self.q: torch.Tensor
        self.register_buffer('q', torch.zeros(1, dtype=torch.int8))
        # q_b is boolean transition state; q remains int8 for downstream bitwise arithmetic.
        #: Boolean complement of :attr:`q`, updated on every call.
        self.q_b: torch.Tensor
        self.register_buffer('q_b', torch.zeros(1, dtype=torch.bool))
        #: Hardware latency and timing metadata for the registered flip-flop output.
        self.hw.pp_delay = 1

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Clear the local output state to ``0`` at every tensor position.
        """
        self.q.resize_(1).zero_()
        self.q_b.resize_(1).zero_()


    def forward(self, input_j: torch.tensor, input_k: torch.tensor):
        """
        Update and return the JK flip-flop state.

        Args:
            input_j: Current 0/1 tensor for the J input.
            input_k: Current 0/1 tensor for the K input.

        Returns:
            The updated Q tensor. The stored state is resized to the broadcast
            result on the first call and retained for the next timestep.

        **Example:**

        .. code-block:: python

            q = flip_flop(torch.tensor([1], dtype=torch.int8),
                          torch.tensor([0], dtype=torch.int8))
        """
        # For Q in {0, 1}, the JK equation reduces to Q' = (not K) if Q else J.
        q_b = torch.where(self.q_b, torch.eq(input_k, 0), torch.ne(input_j, 0))
        if self.q_b.shape == q_b.shape:
            self.q_b.copy_(q_b.detach())
        else:
            self.q_b.resize_as_(q_b).copy_(q_b.detach())
        q = self.q_b.type(torch.int8)
        if self.q.shape == q.shape:
            self.q.copy_(q.detach())
        else:
            self.q.resize_as_(q).copy_(q.detach())
        return self.q.type(self.stype)

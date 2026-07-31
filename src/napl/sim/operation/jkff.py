import torch

from napl.utils import *
from napl.sim.base import napl_base, hw_params


class jkff(napl_base):
    """
    Apply the JK flip-flop state transition to tensor inputs.

    Use this streaming state element when each tensor position needs an
    independent JK flip-flop. ``J=1, K=0`` sets, ``J=0, K=1`` clears, and
    ``J=K=1`` toggles the stored bit.

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
        super().__init__(config, [], polarity_required=False)
        self.hw = hw_params(pp_delay=1)

        self.register_buffer('q', torch.zeros(1, dtype=torch.int8))
        # bool mirror of q; the select consumes this directly so the hot path
        # skips a per-timestep int8->bool cast (q stays int8 for dependents,
        # e.g. sqrt_tracejkff's `(1 - trace) & ...`).
        self.register_buffer('q_b', torch.zeros(1, dtype=torch.bool))


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
        # JK characteristic eq: Q' = (J AND NOT Q) OR (NOT K AND Q). The two
        # terms are mutually exclusive, so for Q in {0,1} this is a plain select:
        # Q' = Q ? (NOT K) : J, a single masked select with no mask or
        # int8-cast temporaries per timestep.
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

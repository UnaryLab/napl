import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import sync_skewed


class min_rc(napl_base):
    r"""
    Select the minimum of two rate-coded streams and track its source.

    The precise target rate-domain operation is

    .. math::

       y = \min(p_0,p_1).

    Let (u_t,v_t) = sync_skewed(input_0,input_1), q_{-1}=0, and q be the
    internal selection state. The output uses the previous state and the
    returned index is 1-q_t:

    .. math::

       \begin{aligned}
       y_t &= x_{1,t}+q_{t-1}(x_{0,t}-x_{1,t}),\\
       q_t &=
       \begin{cases}
       v_t, & u_t\ne v_t,\\
       q_{t-1}, & u_t=v_t.
       \end{cases}
       \end{aligned}

    The returned index is one for input_0 and zero for input_1.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import min_rc

        minimum = min_rc()
        spike, index = minimum(torch.tensor([0], dtype=torch.int8),
                               torch.tensor([1], dtype=torch.int8))
    """


    def __init__(
            self,
            config = {}
    ):
        """
        Construct the selector with its fixed-width synchronizer.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping with no operation-specific keys.
              **name** may optionally label the instance; the default is ``{}``.
        """
        super().__init__(config, [], polarity_required=False)

        #: Previous selection decision used to route the synchronized minimum stream.
        self.index: torch.Tensor
        self.register_buffer('index', torch.zeros(1, dtype=torch.int8))
        #: Skew synchronizer that correlates the two input streams before selection.
        self.sync = sync_skewed({'width': 2})
        #: Hardware latency and timing metadata for the combinational minimum output.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the local comparison state to its initial value.
        """
        self.index.resize_(1).zero_()


    def forward(self, input_0, input_1):
        """
        Select one output spike and update the running argmin.

        Args:
            input_0: Current 0/1 spikes from the first stream.
            input_1: Current 0/1 spikes from the second stream.

        Returns:
            A pair ``(output, index)``. ``output`` is selected using the prior
            state, while ``index`` is the updated state where ``1`` selects the
            first input and ``0`` selects the second.

        **Example:**

        .. code-block:: python

            spike, index = minimum(torch.tensor([0], dtype=torch.int8),
                                   torch.tensor([1], dtype=torch.int8))
        """
        sync_0, sync_1 = self.sync(input_0, input_1)
        sync_0_i8 = sync_0.type(torch.int8)
        sync_1_i8 = sync_1.type(torch.int8)
        d_enable = sync_0_i8 ^ sync_1_i8

        # Output uses the prior selection state; the returned index uses the updated state.
        output = input_1 + self.index * (input_0 - input_1)

        if self.index.shape == d_enable.shape:
            self.index.add_(d_enable * (sync_1_i8 - self.index))
        else:
            updated = self.index + d_enable * (sync_1_i8 - self.index)
            self.index.resize_as_(updated).copy_(updated.detach())

        return output.type(self.stype), 1 - self.index.type(self.stype)

import torch
import math

from napl.sim.base import napl_base, hw_params
from napl.sim.module import gen_num_seq
from loguru import logger


class div_cordiv(napl_base):
    """
    Divide synchronized unipolar streams by correlated division.

    Use this kernel when the dividend and divisor have already been correlated,
    for example by :class:`napl.sync_skewed`. It buffers recent quotient spikes
    and reuses them when the divisor does not spike.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import div_cordiv

        divider = div_cordiv({'depth': 2, 'generator': 'Sobol'})
        quotient = divider(torch.tensor([1], dtype=torch.int8),
                           torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *Design of Division Circuits for Stochastic Computing*.

        *In-Stream Stochastic Division and Square Root via Correlation*.

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*.
    """


    def __init__(
        self,
        config={
            'depth' : 2,
            'generator' : 'Sobol',
        }
    ):
        """
        Configure quotient-history sampling.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **depth**: Number of recent quotient spikes to buffer. It must be a power of two; the default is ``2``.
              - **generator**: Number-sequence generator used to select buffered values; the default is ``"Sobol"``.
              - **dim**: Generator dimension forwarded when the selection sequence is built; the default is ``1``.
              - **seed**: Optional LFSR seed used when **generator** is ``"lfsr"``; the default is ``None``.
              - **taps**: Optional LFSR feedback taps used when **generator** is ``"lfsr"``; the default is ``None``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['depth', 'generator'], polarity_required=False)
        #: Hardware latency and timing metadata for the correlated divider.
        self.hw = hw_params(pp_delay=0)

        #: Number of recent quotient spikes retained for reuse.
        self.depth = config['depth']
        assert math.log2(self.depth) == math.ceil(math.log2(self.depth)), logger.error(f'Input depth <{self.depth}> is not power of 2.')
        #: Number-sequence width needed to address :attr:`depth` history rows.
        self.width = int(math.log2(self.depth))
        config['width'] = self.width

        #: Periodic tensor of quotient-history row selections.
        self.rand_seq: torch.Tensor
        self.register_buffer('rand_seq', torch.floor(gen_num_seq(config).mul(self.depth)).type(torch.long))
        # Python scalar indices avoid device synchronization on each timestep.
        #: Python-list view of :attr:`rand_seq` used for per-timestep indexing.
        self.rand_seq_idx = self.rand_seq.tolist()
        #: Current position in :attr:`rand_seq_idx`.
        self.idx = 0

        #: Recent quotient history, updated only where the divisor spikes.
        self.buffer_q: torch.Tensor
        self.register_buffer('buffer_q', torch.arange(self.depth, dtype=self.stype).remainder_(2))

        #: Whether :attr:`buffer_q` must be expanded to the input shape.
        self.is_first_call = True


    def _reset(self):
        """
        Clear the quotient history and restart its selection sequence.
        """
        self.idx = 0
        self.buffer_q.resize_(self.depth)
        for row in range(self.depth):
            self.buffer_q[row].fill_(row % 2)
        self.is_first_call = True


    def forward(self, dividend, divisor):
        """
        Divide one timestep of synchronized unipolar spikes.

        Args:
            dividend: Current 0/1 dividend spike tensor.
            divisor: Current 0/1 divisor spike tensor, broadcast-compatible with
                ``dividend``.

        Returns:
            A quotient spike tensor with the dividend shape. The call updates
            the sampled quotient history and advances its sequence index.

        **Example:**

        .. code-block:: python

            quotient = divider(torch.tensor([1], dtype=torch.int8),
                               torch.tensor([1], dtype=torch.int8))
        """
        if self.is_first_call:
            dividend_shape = list(dividend.shape)
            divisor_shape = list(divisor.shape)
            if len(dividend_shape) > len(divisor_shape):
                input_shape = dividend_shape
            else:
                input_shape = divisor_shape
            input_shape.insert(0, self.depth)
            self.buffer_q.resize_(input_shape)
            for row in range(self.depth):
                self.buffer_q[row].fill_(row % 2)
            self.is_first_call = False

        divisor_eq_1 = torch.eq(divisor, 1)
        rand_q = self.buffer_q[self.rand_seq_idx[self.idx]]
        self.idx = (self.idx + 1) % self.depth

        quotient = torch.where(divisor_eq_1, dividend, rand_q).view(dividend.size())

        # Shift low to high so each row reads the next history row before it changes.
        buf = self.buffer_q
        for row in range(self.depth - 1):
            torch.where(divisor_eq_1, buf[row + 1], buf[row], out=buf[row])
        torch.where(divisor_eq_1, quotient, buf[-1], out=buf[-1])

        return quotient

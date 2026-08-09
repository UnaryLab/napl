import torch
import math

from napl.sim.base import napl_base
from .encode import encode
from loguru import logger


class div_cordiv(napl_base):
    r"""
    Divide synchronized unipolar streams by correlated division.

    Use this kernel when the dividend and divisor have already been correlated,
    for example by :class:`napl.sync_skewed`.

    The target rate-domain operation is

    .. math::

       y = \frac{x}{d}.

    Both streams are unipolar and must be positively correlated; the quotient
    approaches the target only under that condition.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import div_cordiv

        divider = div_cordiv({'depth': 2, 'generator': 'Sobol'})
        quotient = divider(torch.tensor([1], dtype=torch.int8),
                           torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *Design of division circuits for stochastic computing*, ISVLSI, 2016.

        *In-Stream Stochastic Division and Square Root via Correlation*, DAC, 2019.

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*, IEEE Design & Test, 2021.
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
        super().__init__(config, ['depth', 'generator'], optional_key_list=['polarity', 'dim', 'seed', 'taps'], polarity_required=False)

        #: Number of recent quotient spikes retained for reuse.
        self.depth = config['depth']
        if math.log2(self.depth) != math.ceil(math.log2(self.depth)):
            message = f'Input depth <{self.depth}> is not power of 2.'
            logger.error(message)
            raise AssertionError(message)
        #: Number-sequence width needed to address :attr:`depth` history rows.
        self.width = int(math.log2(self.depth))

        # The encoder supplies the sequence once and is not retained.
        reference_encode = encode({'polarity': 'unipolar',
                                   'timestep': self.depth,
                                   'generator': config['generator'],
                                   'dim': config.get('dim', 1),
                                   'seed': config.get('seed', None),
                                   'taps': config.get('taps', None)})
        #: Periodic tensor of quotient-history row selections.
        self.rand_seq: torch.Tensor
        self.register_buffer('rand_seq',
            reference_encode.num_seq.mul(self.depth).type(torch.long))
        # Python scalar indices avoid device synchronization on each timestep.
        #: Python-list view of :attr:`rand_seq` used for per-timestep indexing.
        self.rand_seq_idx = self.rand_seq.tolist()

        #: Recent quotient history, updated only where the divisor spikes.
        self.buffer_q: torch.Tensor
        self.register_buffer('buffer_q', torch.arange(self.depth, dtype=self.stype).remainder_(2))

        #: Whether :attr:`buffer_q` must be expanded to the input shape.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the correlated divider.
        self.hw.pp_delay = 0

        self.encoding_io = {'dividend': 'rc', 'divisor': 'rc', 'quotient': 'rc'}
        self.polarity_io = {'dividend': 'unipolar', 'divisor': 'unipolar', 'quotient': 'unipolar'}
        self.correlation_i = {('dividend', 'divisor'): 'pos'}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Clear the quotient history and restore its initial shape and values.
        """
        self.buffer_q.resize_(self.depth)
        for row in range(self.depth):
            self.buffer_q[row].fill_(row % 2)
        self.is_first_call = True


    def forward(self, dividend, divisor):
        """
        Divide one timestep of synchronized unipolar spikes with broadcast-compatible inputs.

        Args:
            dividend: Current 0/1 dividend spike tensor.
            divisor: Current 0/1 divisor spike tensor broadcastable against
                ``dividend``.

        Returns:
            A quotient spike tensor with the broadcast shape of the inputs. The
            call updates the quotient history, whose sampled row is given by
            ``rand_seq_idx`` indexed by ``timestep_cur``.
            The history shape is fixed by the first call after ``reset()``; a
            different input shape requires ``reset()``.

        **Example:**

        .. code-block:: python

            quotient = divider(torch.tensor([1], dtype=torch.int8),
                               torch.tensor([1], dtype=torch.int8))
        """
        if self.is_first_call:
            input_shape = torch.broadcast_shapes(dividend.shape, divisor.shape)
            self.buffer_q.resize_((self.depth, *input_shape))
            for row in range(self.depth):
                self.buffer_q[row].fill_(row % 2)
            self.is_first_call = False

        divisor_eq_1 = torch.eq(divisor, 1)
        rand_q = self.buffer_q[self.rand_seq_idx[(self.timestep_cur - 1) % self.depth]]

        quotient = torch.where(divisor_eq_1, dividend, rand_q)

        # Shift low to high so each row reads the next history row before it changes.
        buf = self.buffer_q
        for row in range(self.depth - 1):
            torch.where(divisor_eq_1, buf[row + 1], buf[row], out=buf[row])
        torch.where(divisor_eq_1, quotient, buf[-1], out=buf[-1])

        return quotient

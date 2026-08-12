import torch

from loguru import logger
from napl.sim.base import napl_base
from .sync import sync


class min_sync(napl_base):
    r"""
    Take the minimum of two rate-coded streams with a synchronizer and an AND gate.

    The target rate-domain operation is

    .. math::

       p_y = \min(p_0,p_1),

    which unipolar reads directly as :math:`y = \min(p_0,p_1)`. The bipolar
    reading :math:`v = 2p - 1` is monotone increasing in the rate, so the same
    circuit returns the minimum of the bipolar values,

    .. math::

       y = \min(v_0,v_1).

    A bare AND gate returns the product :math:`p_0 p_1` on uncorrelated streams,
    which lies below the smaller input. This kernel first passes both streams
    through :class:`~napl.sim.operation.sync`, so the ones of the smaller
    stream are covered by the ones of the larger one and the AND gate passes at
    most the ones of the smaller stream.

    The kernel is correlation agnostic at its inputs, since the synchronizer
    supplies the positive correlation the AND gate needs.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import min_sync

        minimum = min_sync({'polarity': 'unipolar', 'depth': 1})
        output = minimum(torch.tensor([1], dtype=torch.int8),
                         torch.tensor([0], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *Correlation Manipulating Circuits for Stochastic Computing*, DATE, 2018.
    """


    def __init__(
            self,
            config={
                'polarity' : 'unipolar',
                'depth' : 1,
            }
    ):
        """
        Configure the stream encoding and the synchronizer save depth.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"unipolar"``.
              - **depth**: Number of unpaired bits the synchronizer can save, an integer of at least ``1``; the default is ``1``. A larger depth raises the accuracy of the minimum.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'depth'], optional_key_list=[], polarity_required=True)

        #: Number of unpaired bits retained by the synchronizer.
        self.depth = config['depth']
        if not isinstance(self.depth, int) or isinstance(self.depth, bool) or self.depth < 1:
            message = f'Invalid depth: <{self.depth}>; legal values: an integer of at least 1.'
            logger.error(message)
            raise AssertionError(message)

        #: Synchronizer that positively correlates the two streams before the AND gate.
        self.sync = sync({'polarity': self.polarity, 'depth': self.depth})
        #: Hardware latency and timing metadata for the combinational minimum output.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_0': self.polarity, 'input_1': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Return the kernel to its initial state.

        The kernel keeps no state of its own beyond the synchronizer, which
        :meth:`napl.sim.base.napl_base.reset` clears as a registered child, so
        this hook does nothing and returns ``None``.
        """
        pass


    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor):
        """
        Process one timestep of the two input streams.

        Args:
            input_0: Current 0/1 spikes from the first stream.
            input_1: Current 0/1 spikes from the second stream.

        Returns:
            A 0/1 spike tensor holding the AND of the two synchronized streams.
            The call also advances the synchronizer state.

        **Example:**

        .. code-block:: python

            output = minimum(torch.tensor([1], dtype=torch.int8),
                             torch.tensor([0], dtype=torch.int8))
        """
        sync_0, sync_1 = self.sync(input_0, input_1)
        return (torch.ne(sync_0, 0) & torch.ne(sync_1, 0)).type(self.stype)

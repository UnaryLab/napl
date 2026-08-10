import torch

from loguru import logger
from napl.sim.base import napl_base
from .desync import desync


class add_desync(napl_base):
    r"""
    Add two unipolar rate-coded streams with a desynchronizer and an OR gate.

    The target rate-domain operation is the saturating sum

    .. math::

       y = \min(1, p_0 + p_1).

    A bare OR gate returns :math:`p_0 + p_1 - p_0 p_1` on uncorrelated streams,
    which falls short of the sum by the overlap of the two streams. This kernel
    first passes both streams through
    :class:`napl.sim.operation.desync`, which moves the ones of the two
    streams apart, so fewer ones collide in the OR gate and the output
    approaches the saturating sum.

    The kernel is correlation agnostic at its inputs, since the desynchronizer
    supplies the negative correlation the OR gate needs.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import add_desync

        adder = add_desync({'polarity': 'unipolar', 'depth': 1})
        output = adder(torch.tensor([1], dtype=torch.int8),
                       torch.tensor([1], dtype=torch.int8))

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
        Configure the stream encoding and the desynchronizer save depth.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding. The only supported value is ``"unipolar"``; the default is ``"unipolar"``.
              - **depth**: Number of paired ones the desynchronizer can save, an integer of at least ``1``; the default is ``1``. A larger depth raises the accuracy of the sum.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'depth'], optional_key_list=[], polarity_required=True)
        if self.polarity != 'unipolar':
            message = f'Invalid polarity: <{self.polarity}>; add_desync supports unipolar only.'
            logger.error(message)
            raise AssertionError(message)

        #: Number of paired ones retained by the desynchronizer.
        self.depth = config['depth']
        if not isinstance(self.depth, int) or isinstance(self.depth, bool) or self.depth < 1:
            message = f'Invalid depth: <{self.depth}>; legal values: an integer of at least 1.'
            logger.error(message)
            raise AssertionError(message)

        #: Desynchronizer that negatively correlates the two streams before the OR gate.
        self.desync = desync({'polarity': 'unipolar', 'depth': self.depth})
        #: Hardware latency and timing metadata for the combinational sum output.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_0': 'unipolar', 'input_1': 'unipolar', 'output': 'unipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Return the kernel to its initial state.

        The kernel keeps no state of its own beyond the desynchronizer, which
        :meth:`napl.sim.base.napl_base.reset` clears as a registered child, so
        this hook does nothing and returns ``None``.
        """
        pass


    def forward(self, input_0: torch.tensor, input_1: torch.tensor):
        """
        Process one timestep of the two input streams.

        Args:
            input_0: Current 0/1 spikes from the first unipolar stream.
            input_1: Current 0/1 spikes from the second unipolar stream.

        Returns:
            A 0/1 spike tensor holding the OR of the two desynchronized streams.
            The call also advances the desynchronizer state.

        **Example:**

        .. code-block:: python

            output = adder(torch.tensor([1], dtype=torch.int8),
                           torch.tensor([1], dtype=torch.int8))
        """
        desync_0, desync_1 = self.desync(input_0, input_1)
        return (torch.ne(desync_0, 0) | torch.ne(desync_1, 0)).type(self.stype)

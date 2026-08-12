import torch

from loguru import logger
from napl.sim.base import napl_base


class subabs(napl_base):
    r"""
    Subtract two unipolar rate-coded streams with a single XOR gate.

    Use this kernel for the absolute difference of two unipolar streams that are
    positively correlated. The target rate-domain operation is

    .. math::

       y = |p_0 - p_1|.

    The kernel is correlation sensitive: it realizes the absolute difference
    only when the two input streams have SCC ``+1``. Decorrelated inputs make
    the output rate approach :math:`p_0 + p_1 - 2 p_0 p_1` instead, so a caller
    that cannot guarantee positively correlated operands passes them through
    :class:`~napl.sim.operation.sync` first.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import subabs

        difference = subabs({'polarity': 'unipolar'})
        output = difference(torch.tensor([1], dtype=torch.int8),
                            torch.tensor([0], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *Correlation Manipulating Circuits for Stochastic Computing*, DATE, 2018.

        *Fast and accurate computation using stochastic circuits*, DATE, 2014.
    """


    def __init__(
            self,
            config={
                'polarity' : 'unipolar',
            }
    ):
        """
        Configure the stream encoding.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding. The only supported value is ``"unipolar"``; the default is ``"unipolar"``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity'], optional_key_list=[], polarity_required=True)
        if self.polarity != 'unipolar':
            message = f'Invalid polarity: <{self.polarity}>; subabs supports unipolar only.'
            logger.error(message)
            raise AssertionError(message)

        #: Hardware latency and timing metadata for the combinational XOR output.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_0': 'unipolar', 'input_1': 'unipolar', 'output': 'unipolar'}
        self.correlation_i = {('input_0', 'input_1'): 'pos'}


    def _reset(self):
        """
        Reset no local state; the kernel keeps nothing across timesteps.
        """
        pass


    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor):
        """
        Process one timestep of the two input streams.

        Args:
            input_0: Current 0/1 spikes from the first unipolar stream.
            input_1: Current 0/1 spikes from the second unipolar stream.

        Returns:
            A 0/1 spike tensor holding the bitwise XOR of the two inputs.

        **Example:**

        .. code-block:: python

            output = difference(torch.tensor([1], dtype=torch.int8),
                                torch.tensor([0], dtype=torch.int8))
        """
        return torch.ne(input_0, input_1).type(self.stype)

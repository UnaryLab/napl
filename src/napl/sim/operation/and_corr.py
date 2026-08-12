import torch

from loguru import logger
from napl.sim.base import napl_base


class and_corr(napl_base):
    r"""
    Minimum of two unipolar rate-coded streams with a single AND gate.

    Use this stateless kernel for the elementwise minimum of two unipolar
    streams that are maximally positively correlated (SCC ``+1``, meaning both
    operands are encoded on the same number sequence / Sobol dimension). The
    target rate-domain operation is

    .. math::

       p_y = \min(p_0, p_1).

    The kernel is correlation sensitive. Bitwise AND realizes the minimum only
    at SCC ``+1``: when the two streams share the same sequence, the shorter
    stream's ones are a subset of the longer stream's ones, so their overlap is
    exactly the smaller rate. On **decorrelated** (independent) inputs the same
    AND gate instead computes the product :math:`p_0 p_1` (the ``mul_gaines``
    behavior), so the correlation requirement is what turns this gate from a
    multiplier into a minimum. A caller that cannot guarantee correlated
    operands passes them through :class:`~napl.sim.operation.sync` first.

    Unipolar only. The AND-as-minimum identity relies on a stream's ones being a
    subset relation, which the [0, 1] unipolar encoding provides; the bipolar
    encoding carries no such subset order, so a bipolar configuration is
    rejected.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import and_corr

        minimum = and_corr({'polarity': 'unipolar'})
        output = minimum(torch.tensor([1], dtype=torch.int8),
                         torch.tensor([0], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        Derived kernel: bitwise AND on maximally-correlated unipolar streams.

        *Correlation Manipulating Circuits for Stochastic Computing*, DATE, 2018.
    """


    def __init__(
            self,
            config={
                'polarity': 'unipolar',
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
        # AND computes the minimum only under the unipolar subset order; the
        # bipolar encoding carries no such order, so it has no AND-minimum form.
        if self.polarity != 'unipolar':
            message = f'Invalid polarity: <{self.polarity}>; and_corr supports unipolar only.'
            logger.error(message)
            raise AssertionError(message)

        #: Hardware latency and timing metadata for the combinational AND output.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_0': 'unipolar', 'input_1': 'unipolar', 'output': 'unipolar'}
        # Maximal positive correlation (SCC +1) between the operands is required
        # for AND to realize the minimum rather than the product.
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
            A 0/1 spike tensor holding the bitwise AND of the two inputs, which
            is the elementwise minimum when the streams are correlated. The
            method changes no local state.

        **Example:**

        .. code-block:: python

            output = minimum(torch.tensor([1], dtype=torch.int8),
                             torch.tensor([0], dtype=torch.int8))
        """
        return (input_0.type(torch.int8) & input_1.type(torch.int8)).type(self.stype)

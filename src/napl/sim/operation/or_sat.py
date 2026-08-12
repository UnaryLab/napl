import torch

from loguru import logger
from napl.sim.base import napl_base


class or_sat(napl_base):
    r"""
    Saturating add of two unipolar rate-coded streams by a bitwise OR gate.

    Use this stateless operation to add two independent unipolar streams with a
    single OR gate: each timestep emits ``a_bit OR b_bit``. On rate-coded values
    the OR gate realizes the probabilistic-OR saturating add

    .. math::

       p_y = p_a + p_b - p_a p_b,

    which stays inside ``[0, 1]`` for any ``p_a, p_b`` in ``[0, 1]`` and so never
    overflows, unlike a plain sum.

    The two operands must be **independent** (decorrelated): the
    ``p_a + p_b - p_a p_b`` value holds only when the streams share no
    correlation, so encode them on distinct number-sequence dimensions.

    Unipolar only. A bitwise OR of two bipolar streams has no clean saturating-add
    value semantics, so a bipolar configuration is rejected.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import or_sat

        saturating_add = or_sat({'polarity': 'unipolar'})
        output = saturating_add(torch.tensor([1, 0], dtype=torch.int8),
                                torch.tensor([0, 1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        Derived kernel: per-timestep bitwise OR of two independent unipolar streams.

        *Stochastic Computing Systems*, Advances in Information Systems Science, 1969.
    """


    def __init__(
            self,
            config={
                'polarity': 'unipolar',
            }
        ):
        """
        Configure the stream polarity.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Must be ``"unipolar"``; the default is ``"unipolar"``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity'], polarity_required=True)
        # A bitwise OR realizes a + b - a*b on unipolar rates; the same gate on
        # bipolar (-1/+1) streams carries no saturating-add value semantics, so
        # only unipolar is admissible.
        if self.polarity != 'unipolar':
            message = f'Invalid polarity: <{self.polarity}>; legal values: <[\'unipolar\']>.'
            logger.error(message)
            raise AssertionError(message)

        # A bare OR gate is combinational, so it adds no pipeline delay.
        #: Hardware latency and timing metadata for the combinational OR gate.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_0': self.polarity, 'input_1': self.polarity, 'output': self.polarity}
        # a + b - a*b holds only for decorrelated operands, so the two inputs must
        # carry zero cross-correlation (distinct number-sequence dimensions).
        self.correlation_i = {('input_0', 'input_1'): 'zero'}


    def _reset(self):
        """
        Reset no local mutable state.
        """
        pass


    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor):
        """
        Saturating-add one timestep of two unipolar streams.

        Args:
            input_0: Current 0/1 spike tensor from the first stream.
            input_1: Current 0/1 spike tensor from the second stream.

        Returns:
            The elementwise OR spike tensor in the configured spike dtype. The
            method changes no local state.

        **Example:**

        .. code-block:: python

            output = saturating_add(torch.tensor([1, 0], dtype=torch.int8),
                                    torch.tensor([0, 1], dtype=torch.int8))
        """
        return (input_0.type(torch.int8) | input_1.type(torch.int8)).type(self.stype)

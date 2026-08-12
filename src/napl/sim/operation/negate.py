import torch
from loguru import logger

from napl.sim.base import napl_base


class negate(napl_base):
    r"""
    Negate a bipolar rate-coded spike stream by inverting its spikes.

    Use this stateless operation wherever a bipolar stream must change sign,
    such as feeding a subtraction path built from an adder. Inverting every
    spike turns a one-density :math:`p_x` into :math:`1 - p_x`, so the bipolar
    value flips sign,

    .. math::

       p_y = 1 - p_x,\qquad v_y = 2 p_y - 1 = -(2 p_x - 1) = -v_x.

    Only bipolar encoding is supported, because a unipolar stream cannot
    represent a negative value; there the complement :math:`1 - p_x` is a
    different operation than a negation.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import negate

        op = negate({'polarity': 'bipolar'})
        output = op(torch.tensor([1, 0], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *Stochastic Computing Systems*, Advances in Information Systems Science, 1969.
    """


    def __init__(
            self,
            config={
                'polarity': 'bipolar',
            }
        ):
        """
        Select the stream polarity.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Must be ``"bipolar"``; the default is ``"bipolar"``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity'], polarity_required=True)
        # A unipolar stream carries no negative value, so its complement is a
        # different operation and only bipolar encoding negates by inversion.
        if self.polarity != 'bipolar':
            message = f'Invalid polarity: <{self.polarity}>; legal values: <[\'bipolar\']>.'
            logger.error(message)
            raise AssertionError(message)
        # Inverting a spike is a single gate, so the path is combinational.
        #: Hardware latency and timing metadata for the combinational inverter.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'bipolar', 'output': 'bipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Reset no local mutable state.
        """
        pass


    def forward(self, input: torch.Tensor):
        """
        Negate one timestep of a bipolar stream.

        Args:
            input: Current 0/1 spike tensor of the bipolar stream.

        Returns:
            The elementwise inverted spike tensor with the same shape, in the
            configured spike dtype. The method changes no local state.

        **Example:**

        .. code-block:: python

            output = op(torch.tensor([1, 0], dtype=torch.int8))
        """
        # bitwise_xor rejects a float input, so the spike is cast to int8 first.
        return input.type(torch.int8).bitwise_xor(1).type(self.stype)

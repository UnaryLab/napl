import torch

from napl.sim.base import napl_base
from .dff import dff


class square_dff(napl_base):
    r"""
    Square a unary stream by multiplying it with a delayed copy.

    The rate-domain operation is

    .. math::

       p_y = p_x^2 \quad (\text{unipolar}),\qquad
       v_y = v_x^2 \quad (\text{bipolar}).

    A **depth**-timestep delay decorrelates the stream from itself. The first
    **depth** outputs are formed against the zero-filled delay line.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import square_dff

        square = square_dff({'polarity': 'unipolar', 'depth': 1})
        output = square(torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.

        *uGEMM: Unary Computing for GEMM Applications*, IEEE Micro, 2021.
    """


    def __init__(
            self,
            config={
                'polarity': 'bipolar',
                'depth': 1,
            }
        ):
        """
        Configure the stream polarity and decorrelation delay.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Use ``"unipolar"`` for AND or ``"bipolar"`` for XNOR; the default is ``"bipolar"``.
              - **depth**: Number of timesteps in the internal D flip-flop delay; the default is ``1``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'depth'], polarity_required=True)

        #: Delay line that supplies the earlier spike multiplied with the current input.
        self.dff = dff(config={'depth': config['depth']})

        # Bitwise operands remain int8; floating spike types are cast to int8.
        self._spike_is_int8 = (self.stype == torch.int8)
        #: Hardware latency and timing metadata for the combinational square output.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Reset no additional local state beyond the internal delay module.
        """
        pass


    def forward(self, input: torch.Tensor):
        """
        Square one timestep against its delayed copy.

        Args:
            input: Current 0/1 spike tensor.

        Returns:
            The elementwise AND or XNOR of the current and delayed spikes. The
            internal D flip-flop stores the current input for later timesteps.

        **Example:**

        .. code-block:: python

            output = square(torch.tensor([1], dtype=torch.int8))
        """
        input_d = self.dff(input)
        if self._spike_is_int8:
            a, b = input, input_d
        else:
            a, b = input.type(torch.int8), input_d.type(torch.int8)
        if self.polarity == 'unipolar':
            out = a & b
        else:
            # For 0/1 spikes, XNOR is 1 - (a ^ b).
            out = 1 - (a ^ b)
        return out if self._spike_is_int8 else out.type(self.stype)

import torch

from loguru import logger
from napl.sim.base import napl_base
from .add_scale import add_scale
from .negate import negate


class sub_scale(napl_base):
    r"""
    Subtract two bipolar rate-coded streams with a configurable output scale.

    Use this streaming kernel for a signed scaled difference of two independent
    bipolar streams. The target rate-domain operation is

    .. math::

       v_y = \frac{v_a - v_b}{\mathit{scale}},

    realized as negate-then-scaled-add: the second stream is negated and the two
    streams are reduced by a scaled adder, so ``(a - b) / scale`` equals
    ``(a + (-b)) / scale``. ``scale`` is a positive fixed-point constant on the
    adder's grid; keep ``scale`` at least ``|v_a - v_b|`` so the difference stays
    inside the representable bipolar range ``[-1, 1]``.

    Bipolar only. Negation inverts spikes to flip a value's sign, which a
    unipolar stream cannot represent, so the negate-then-add mechanism has no
    unipolar form and a unipolar configuration is rejected.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import sub_scale

        difference = sub_scale({'polarity': 'bipolar', 'scale': 2, 'intwidth': 20, 'fracwidth': 4})
        output = difference(torch.tensor([1, 0], dtype=torch.int8),
                            torch.tensor([0, 1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        Derived kernel: negate composed with add_scale.

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """


    def __init__(
            self,
            config={
                'polarity': 'bipolar',
                'scale': 2,
                'intwidth': 10,
                'fracwidth': 0,
            }
        ):
        """
        Configure the polarity and the scaled adder's fixed-point format.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Must be ``"bipolar"``; the default is ``"bipolar"``.
              - **scale**: Amount of ``(a - b)`` consumed per output spike, a positive finite number quantized to the adder's grid; the default is ``2``.
              - **intwidth**: Integer bits of the signed accumulator, including the sign bit; the default is ``10``.
              - **fracwidth**: Fractional bits of the signed accumulator; the default is ``0``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'scale', 'intwidth', 'fracwidth'], polarity_required=True)
        # The mechanism negates the second stream, and a unipolar stream carries no
        # negative value to negate, so only bipolar is admissible.
        if self.polarity != 'bipolar':
            message = f'Invalid polarity: <{self.polarity}>; legal values: <[\'bipolar\']>.'
            logger.error(message)
            raise AssertionError(message)

        # Gate 18 reuse: hold the negate and add_scale operations instead of
        # reinlining their math. negate flips the second stream's sign; add_scale
        # reduces the two streams onto the output grid. add_scale validates scale,
        # intwidth, and fracwidth on construction.
        #: Sign-flip stage applied to the second input stream.
        self.negate = negate({'polarity': 'bipolar'})
        #: Scaled two-input adder that emits the difference on the output grid.
        self.add_scale = add_scale({
            'polarity': 'bipolar',
            'scale': config['scale'],
            'intwidth': config['intwidth'],
            'fracwidth': config['fracwidth'],
        })

        #: Amount of ``(a - b)`` consumed per output spike, quantized to the grid.
        self.scale = self.add_scale.scale
        # The path is negate (combinational) feeding add_scale (combinational), so
        # its pipeline delay is the sum of the two composed delays.
        #: Hardware latency and timing metadata for the composed difference path.
        self.hw.pp_delay = self.negate.hw.pp_delay + self.add_scale.hw.pp_delay

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_0': 'bipolar', 'input_1': 'bipolar', 'output': 'bipolar'}
        self.correlation_i = {}


    def _reset(self):
        """
        Reset no local state; the composed negate and add_scale are reset by
        :meth:`reset` before this local reset.
        """
        pass


    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor):
        """
        Process one timestep of the two bipolar input streams.

        Args:
            input_0: Current 0/1 spikes of the minuend stream ``a``.
            input_1: Current 0/1 spikes of the subtrahend stream ``b``.

        Returns:
            A 0/1 spike tensor of the scaled difference ``(a - b) / scale`` with
            the same shape as each input. The call advances the composed adder by
            one timestep.

        **Example:**

        .. code-block:: python

            output = difference(torch.tensor([1, 0], dtype=torch.int8),
                                torch.tensor([0, 1], dtype=torch.int8))
        """
        # Stack a and (-b) on a new trailing axis so the scaled adder reduces the
        # two streams into (a + (-b)) / scale.
        negated = self.negate(input_1)
        stacked = torch.stack((input_0, negated), dim=-1)
        return self.add_scale(stacked, dim=-1)

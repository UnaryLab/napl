import torch

from napl.sim.base import napl_base


class mux_select(napl_base):
    r"""
    Select between two spike streams with a data-driven multiplexer.

    Use this stateless operation to route one of two rate-coded streams through
    to the output under the control of a third rate-coded *select* stream, the
    per-timestep ``c ? a : b``. The select stream carries data, not a fair coin,
    so its rate sets the mixing weight between the two operands.

    The target rate-domain operation, valid in both polarities, is

    .. math::

       p_y = p_c p_a + (1 - p_c) p_b \quad (\text{unipolar}),\qquad
       v_y = p_c v_a + (1 - p_c) v_b \quad (\text{bipolar}),

    where :math:`p_c` is the select stream's rate in ``[0, 1]``. The select
    stream carries this mixing weight as a rate, so it is always unipolar; the
    two operands and the output take the configured polarity.

    The select stream must be decorrelated from both operands; encode the three
    inputs on distinct number-sequence dimensions. Correlation between the two
    operands does not affect the result, since the output never combines them.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import mux_select

        select = mux_select({'polarity': 'bipolar'})
        output = select(torch.tensor([1], dtype=torch.int8),
                        torch.tensor([1], dtype=torch.int8),
                        torch.tensor([0], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *2-to-1 stream multiplexer with a data select*, derived.
    """


    def __init__(
            self,
            config={
                'polarity': 'bipolar',
            }
        ):
        """
        Set the stream polarity for the multiplexer.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity'], polarity_required=True)
        # The 2-to-1 select is a combinational gate over the three current spikes.
        #: Hardware latency and timing metadata for the combinational multiplexer.
        self.hw.pp_delay = 0

        self.encoding_io = {'select': 'rc', 'input_a': 'rc', 'input_b': 'rc', 'output': 'rc'}
        # The select stream carries the mixing weight p_c as a rate, so it is
        # always unipolar; only the operands and the output follow the polarity.
        self.polarity_io = {
            'select': 'unipolar',
            'input_a': self.polarity,
            'input_b': self.polarity,
            'output': self.polarity,
        }
        # E[out] = p_c p_a + p_b - p_c p_b holds the value semantics only when the
        # select is independent of each operand; the two operands need no mutual
        # constraint, since the output never mixes them in one term.
        self.correlation_i = {('select', 'input_a'): 'zero', ('select', 'input_b'): 'zero'}


    def _reset(self):
        """
        Reset no local mutable state.
        """
        pass


    def forward(self, select: torch.Tensor, input_a: torch.Tensor, input_b: torch.Tensor):
        """
        Select one timestep from two streams under the select stream.

        Args:
            select: Current 0/1 spike tensor steering the multiplexer.
            input_a: Current 0/1 spike tensor routed through when ``select`` is 1.
            input_b: Current 0/1 spike tensor routed through when ``select`` is 0.

        Returns:
            The elementwise selected spike tensor in the configured spike dtype.
            The method changes no local state.

        **Example:**

        .. code-block:: python

            output = select(torch.tensor([1], dtype=torch.int8),
                            torch.tensor([1], dtype=torch.int8),
                            torch.tensor([0], dtype=torch.int8))
        """
        # Bitwise select out = c * a + (1 - c) * b, polarity-agnostic on the bits.
        return torch.where(select.type(torch.int8).bool(), input_a, input_b).type(self.stype)

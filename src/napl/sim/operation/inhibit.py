import torch

from napl.sim.base import napl_base, hw_params


class inhibit(napl_base):
    r"""
    Gate a temporal-coded data stream with an inhibiting stream (race-logic
    INHIBIT).

    Race logic encodes a value as the arrival time of an edge. INHIBIT passes
    the data signal unchanged when it arrives before or at the same time as the
    inhibiting signal, and never falls when the inhibiting signal arrives
    strictly earlier. napl temporal streams carry the value on a falling edge
    that comes later for larger values, so the inhibiting value blocks the data
    value exactly when it is strictly smaller, and the never-falling all-ones
    output decodes to the maximum representable value. For data value
    :math:`x_0` and inhibiting value :math:`x_1`

    .. math::

       y =
       \begin{cases}
       x_0, & x_0 \le x_1,\\
       y_{\max}, & x_0 > x_1,
       \end{cases}

    with :math:`y_{\max}=1` for unipolar and :math:`y_{\max}=+1` for bipolar
    streams.

    Relative to Tzimpragos et al. (2019): napl shares that paper's value-to-time
    direction, value = arrival time with a larger value arriving later, and
    carries it on a falling rather than a rising edge. Complementing both inputs
    and the output recovers the paper's rising-edge gate exactly, so the pass
    condition :math:`x_0 \le x_1` above is the paper's :math:`j \le i` with no
    mirroring, ``input_data`` being the data signal :math:`j` and ``input_inhibit`` the
    inhibiting signal :math:`i`. This is the paper's Fig. 3(c) synchronous form,
    in which a simultaneous arrival passes, rather than the Fig. 3(d)
    asynchronous latch realizing strict :math:`j < i`.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import inhibit

        gate = inhibit()
        output = gate(torch.tensor([1], dtype=torch.int8),
                      torch.tensor([0], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *Race Logic: A hardware acceleration for dynamic programming algorithms*, ISCA, 2014.

        *Space-Time Computing with Temporal Neural Networks*, Synthesis Lectures on Computer Architecture, 2017.

        *Space-Time Algebra: A Model for Neocortical Computation*, ISCA, 2018.

        *Boosted Race Trees for Low Energy Classification*, ASPLOS, 2019.
    """


    def __init__(
            self,
            config = {}
    ):
        """
        Construct the temporal-code INHIBIT gate.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping with no operation-specific keys.
              **name** may optionally label the instance; the default is ``{}``.
        """
        super().__init__(config, [], optional_key_list=['polarity'], polarity_required=False)

        #: Latched inhibition state; ``1`` holds the output high permanently.
        self.latch: torch.Tensor
        self.register_buffer('latch', torch.zeros(1, dtype=torch.int8))
        #: Hardware latency and timing metadata for the combinational INHIBIT output.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input_data': 'tc', 'input_inhibit': 'tc', 'output': 'tc'}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the local inhibition latch to the uninhibited state.
        """
        self.latch.resize_(1).zero_()


    def forward(self, input_data, input_inhibit):
        """
        Compute one timestep of the temporal-code INHIBIT gate.

        Args:
            input_data: Current 0/1 tensor from the data temporal stream.
            input_inhibit: Current 0/1 tensor from the inhibiting temporal stream.

        Returns:
            The data spike held high by the inhibition latch. The latch is set
            permanently once the data stream is still high while the
            inhibiting stream has already fallen, after which the output stays
            high for the rest of the codeword.

        **Example:**

        .. code-block:: python

            output = gate(torch.tensor([1], dtype=torch.int8),
                          torch.tensor([0], dtype=torch.int8))
        """
        in_data = input_data.type(torch.int8)
        in_inhibit = input_inhibit.type(torch.int8)
        blocked = in_data & (in_inhibit ^ 1)

        if self.latch.shape == blocked.shape:
            self.latch.bitwise_or_(blocked)
        else:
            updated = self.latch | blocked
            self.latch.resize_as_(updated).copy_(updated.detach())

        output = in_data | self.latch
        return output.type(self.stype)

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
    streams. Per timestep, a latch records that the data stream was still high
    while the inhibiting stream had fallen, with :math:`s_{-1}=0`:

    .. math::

       \begin{aligned}
       s_t &= s_{t-1}\mathbin{\lor}(x_{0,t}\mathbin{\land}\lnot x_{1,t}),\\
       y_t &= x_{0,t}\mathbin{\lor}s_t.
       \end{aligned}

    Relative to Tzimpragos et al. (2019): napl shares that paper's value-to-time
    direction, value = arrival time with a larger value arriving later, and
    carries it on a falling rather than a rising edge. Complementing both inputs
    and the output recovers the paper's rising-edge gate exactly, so the pass
    condition :math:`x_0 \le x_1` above is the paper's :math:`j \le i` with no
    mirroring, ``input_0`` being the data signal :math:`j` and ``input_1`` the
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

    .. rubric:: References

    A. Madhavan, T. Sherwood, and D. Strukov, *Race Logic: A Hardware
    Acceleration for Dynamic Programming Algorithms*, ISCA, 2014. (introduces
    race logic: MAX, MIN, ADD-CONSTANT)
    https://sites.cs.ucsb.edu/~sherwood/pubs/ISCA-14-racelogic.pdf

    J. E. Smith, *Space-Time Computing with Temporal Neural Networks*,
    Synthesis Lectures on Computer Architecture, 2017. (temporal spike
    computing that motivates INHIBIT)
    https://link.springer.com/book/10.1007/978-3-031-01754-4

    J. E. Smith, *Space-Time Algebra: A Model for Neocortical Computation*,
    ISCA, 2018. (space-time algebra generalizing race logic with inhibition)
    https://ieeexplore.ieee.org/document/8416835

    G. Tzimpragos, A. Madhavan, D. Vasudevan, D. Strukov, and T. Sherwood,
    *Boosted Race Trees for Low Energy Classification*, ASPLOS, 2019. (adds
    INHIBIT to the race-logic primitive set)
    https://sites.cs.ucsb.edu/~sherwood/pubs/ASPLOS-19-racetree.pdf
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
        super().__init__(config, [], polarity_required=False)

        #: Latched inhibition state; ``1`` forces the output low permanently.
        self.latch: torch.Tensor
        self.register_buffer('latch', torch.zeros(1, dtype=torch.int8))
        #: Hardware latency and timing metadata for the combinational INHIBIT output.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input_0': 'tc', 'input_1': 'tc', 'output': 'tc'}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restore the local inhibition latch to the uninhibited state.
        """
        self.latch.resize_(1).zero_()


    def forward(self, input_0, input_1):
        """
        Compute one timestep of the temporal-code INHIBIT gate.

        Args:
            input_0: Current 0/1 tensor from the data temporal stream.
            input_1: Current 0/1 tensor from the inhibiting temporal stream.

        Returns:
            The data spike masked by the inhibition latch. The latch is set
            permanently once the inhibiting stream spikes while the data
            stream has not.

        **Example:**

        .. code-block:: python

            output = gate(torch.tensor([1], dtype=torch.int8),
                          torch.tensor([0], dtype=torch.int8))
        """
        in_0 = input_0.type(torch.int8)
        in_1 = input_1.type(torch.int8)
        blocked = in_0 & (in_1 ^ 1)

        if self.latch.shape == blocked.shape:
            self.latch.bitwise_or_(blocked)
        else:
            updated = self.latch | blocked
            self.latch.resize_as_(updated).copy_(updated.detach())

        output = in_0 | self.latch
        return output.type(self.stype)

import torch
import math

from napl.sim.base import napl_base
from loguru import logger


class decode(napl_base):
    r"""Decode a unary spike stream into a progressive numeric value.

    Use this module when a streaming simulation needs the running unipolar or
    bipolar value after each timestep. Access :attr:`spike_value` after feeding
    one or more spike tensors.

    The target inverts the encoding, mapping the rate of the first :math:`T`
    spikes back to a numeric value,

    .. math::

       \hat x_T = \begin{cases}
       c_T / T, & \text{unipolar},\\
       2 c_T / T - 1, & \text{bipolar},
       \end{cases}
       \qquad c_T = \sum_{t=1}^{T} s_t.

    The value is zero before the first timestep.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import decode

        dec = decode({"polarity": "unipolar", "timestep": 4})
        dec(torch.tensor([1.0, 0.0]))
        value = dec.spike_value
    """


    def __init__(
            self,
            config:dict={
                'polarity': 'bipolar',
                'timestep': 256,
                }
        ):
        """Configure the stream representation and maximum run length.

        Args:
            config: Configuration mapping with these keys:

                * **polarity** - ``"unipolar"`` or ``"bipolar"``. Defaults to
                  ``"bipolar"``.
                * **timestep** - Maximum number of input timesteps. Defaults to
                  ``256``.
                * **name** - Optional instance label. Defaults to ``None``.

        Construction initializes the spike counter to a scalar zero.
        """
        super().__init__(config, ['polarity', 'timestep'], optional_key_list=['generator', 'dim', 'seed', 'taps'], polarity_required=True)

        #: Maximum number of spike timesteps accepted by this decoder.
        self.timestep = config['timestep']
        #: Bit width needed to count through the configured stream length.
        self.width = math.ceil(math.log2(self.timestep))

        #: Per-element count of received one-valued spikes.
        self.spike_count: torch.Tensor
        self.register_buffer('spike_count', torch.zeros(1, dtype=self.ntype))

        self.hw.pp_delay = 0

        self.encoding_io = {}
        self.polarity_io = {'input': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Clear the locally accumulated spike count.

        The count returns to a scalar zero and expands to the next input shape on
        demand. This hook returns ``None`` and is called by ``reset()``.
        """
        self.spike_count.resize_(1).zero_()


    def forward(self, input: torch.Tensor):
        """Accumulate one spike tensor.

        Args:
            input: Current ``0``/``1`` spike tensor. Its shape becomes the
                decoder state shape on the first call.

        Returns:
            ``None``. Read :attr:`spike_value` for the progressive decoded value.

        The call adds ``input`` to ``spike_count`` and advances
        ``timestep_cur``. It rejects calls beyond the configured **timestep**.
        """
        if self.timestep_cur > self.timestep:
            message = f'Timestep <{self.timestep_cur}> exceeds the maximum timestep <{self.timestep}>.'
            logger.error(message)
            raise AssertionError(message)
        sc = self.spike_count
        # The scalar seed broadcasts once; matching shapes then accumulate in place.
        if sc.shape == input.shape:
            sc.add_(input)
        else:
            expanded = sc.add(input).detach()
            self.spike_count.resize_as_(expanded).copy_(expanded)


    @property
    def spike_value(self):
        """Return the progressive value represented by the received spikes.

        Returns:
            A fresh tensor equal to ``spike_count / timestep_cur`` for unipolar
            streams or ``2 * spike_count / timestep_cur - 1`` for bipolar
            streams. Before the first call, it returns a zero tensor.

        Reading the property does not change the decoder state.

        **Example:**

        .. code-block:: python

            dec.reset()
            dec(torch.ones(2))
            assert torch.equal(dec.spike_value, torch.ones(2))
        """
        if self.timestep_cur == 0:
            return torch.zeros_like(self.spike_count)
        # Division returns a fresh tensor, so bipolar rescaling cannot alter the count.
        sv = self.spike_count.div(self.timestep_cur)
        if self.polarity == 'bipolar':
            sv.mul_(2).sub_(1)
        return sv

import torch
import math

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger


class decoder(napl_base):
    """Decode a unary spike stream into a progressive numeric value.

    Use this module when a streaming simulation needs the running unipolar or
    bipolar value after each timestep. Access :attr:`spike_value` after feeding
    one or more spike tensors.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import decoder

        dec = decoder({"polarity": "unipolar", "timestep": 4})
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
        super().__init__(config, ['polarity', 'timestep'], polarity_required=True)

        # initialize timestep and spike count
        self.timestep = config['timestep']
        self.width = math.ceil(math.log2(self.timestep))

        self.register_buffer('spike_count', torch.zeros(1, dtype=self.ntype))


    def _reset(self):
        """Clear the locally accumulated spike count.

        The count returns to a scalar zero and expands to the next input shape on
        demand. This hook returns ``None`` and is called by ``reset()``.
        """
        self.spike_count.resize_(1).zero_()


    def forward(self, spike: torch.Tensor):
        """Accumulate one spike tensor.

        Args:
            spike: Current ``0``/``1`` spike tensor. Its shape becomes the
                decoder state shape on the first call.

        Returns:
            ``None``. Read :attr:`spike_value` for the progressive decoded value.

        The call adds ``spike`` to ``spike_count`` and advances
        ``timestep_cur``. It rejects calls beyond the configured **timestep**.
        """
        # get the spike value at the current timestep
        assert self.timestep_cur <= self.timestep, \
            logger.error(f'Timestep <{self.timestep_cur}> exceeds the maximum timestep <{self.timestep}>.')
        # float accumulator avoids overflow; the 0/1 spike promotes exactly, so no cast.
        sc = self.spike_count
        # shape-guarded: first forward broadcasts the (1,) seed up to spike's shape
        # out-of-place; steady state accumulates in place to drop a per-timestep alloc.
        if sc.shape == spike.shape:
            sc.add_(spike)
        else:
            expanded = sc.add(spike).detach()
            self.spike_count.resize_as_(expanded).copy_(expanded)
        # no return: evaluating the spike_value property here would redo the div
        # every timestep; readers access .spike_value on demand instead.


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

            dec(torch.ones(2))
            assert torch.equal(dec.spike_value, torch.ones(2))
        """
        if self.timestep_cur == 0:
            return torch.zeros_like(self.spike_count)
        # sv is the fresh div result, so the in-place bipolar rescale leaves spike_count untouched.
        sv = self.spike_count.div(self.timestep_cur)
        if self.polarity == 'bipolar':
            sv.mul_(2).sub_(1)
        return sv

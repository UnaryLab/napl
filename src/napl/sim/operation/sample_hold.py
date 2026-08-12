import torch

from napl.sim.base import napl_base
from loguru import logger
from .decode import decode
from .encode import encode


class sample_hold(napl_base):
    r"""
    Freeze a stream's decoded estimate on a trigger and re-emit it as a constant.

    Use this streaming kernel to bridge progressive precision to value reuse: it
    watches a unipolar or bipolar rate-coded input, keeps a running decoded
    estimate of its value, and on a configured trigger latches that estimate and
    thereafter emits a fresh spike stream encoding the frozen value, ignoring all
    later input.

    The trigger is the fixed sample timestep ``trigger_timestep`` :math:`= k`. It
    partitions one run into two phases, with the latch value equal to the decoded
    estimate over the first :math:`k` timesteps,

    .. math::

       \hat x_k = \begin{cases}
       c_k / k, & \text{unipolar},\\
       2 c_k / k - 1, & \text{bipolar},
       \end{cases}
       \qquad c_k = \sum_{t=1}^{k} s_t.

    For timesteps :math:`t < k` the call passes the live input spike through
    unchanged. At timestep :math:`t = k` it latches :math:`\hat x_k`. For every
    timestep :math:`t \ge k` it returns a spike re-encoding the latched value, so
    the output no longer depends on the input. The frozen value is unchanged by
    later input until :meth:`reset`, which clears the latch and starts a new run.
    Both polarities are supported, since freezing a decoded value is independent
    of the encoding.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import sample_hold

        operation = sample_hold({'polarity': 'unipolar', 'timestep': 256,
                                 'trigger_timestep': 128})
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *Sample-and-hold reuse of a progressively decoded unary stream*, derived.
    """


    def __init__(
        self,
        config={
            'polarity': 'bipolar',
            'timestep': 256,
        },
    ):
        """
        Configure the stream encoding, run length, and trigger timestep.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **timestep**: Maximum run length, shared by the internal decoder and re-encoder; the default is ``256``.
              - **trigger_timestep**: One-based timestep at which the running estimate is latched. It must lie in ``[1, timestep]``; the default is ``timestep // 2``.
              - **generator**: Number-sequence generator for the re-emitted stream; the default is ``"sobol"``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['polarity', 'timestep'], optional_key_list=['trigger_timestep', 'generator'], polarity_required=True)

        #: Maximum run length shared by the internal decoder and re-encoder.
        self.timestep = config['timestep']
        #: One-based timestep at which the running estimate is latched.
        self.trigger_timestep = config.get('trigger_timestep', self.timestep // 2)
        if not (1 <= self.trigger_timestep <= self.timestep):
            message = f'Invalid trigger_timestep: <{self.trigger_timestep}>; legal values: an integer in [1, {self.timestep}].'
            logger.error(message)
            raise AssertionError(message)

        codec = {'polarity': self.polarity, 'timestep': self.timestep}
        #: Progressive decoder tracking the running estimate of the live stream.
        self.decoder = decode(codec)
        #: Re-encoder that emits the frozen value as a fresh spike stream.
        self.reference_encode = encode({**codec, 'generator': config.get('generator', 'sobol')})

        #: Latched decoded estimate re-emitted after the trigger.
        self.held: torch.Tensor
        self.register_buffer('held', torch.zeros(1, dtype=self.ntype))
        #: Whether the estimate has been latched in the current run.
        self.latched = False

        # The kernel is a behavioral sim composition with no gate-level RTL, so it declares no pipeline latency.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}


    def _reset(self):
        """
        Clear the latched value and drop back to the pre-trigger phase.

        The internal decoder and re-encoder are reset by :meth:`reset` before this
        local reset. This hook returns ``None``.
        """
        self.held.resize_(1).zero_()
        self.latched = False


    def forward(self, input):
        """
        Process one timestep of a rate-coded input stream.

        Before the trigger the call passes ``input`` through; at the trigger it
        latches the running decoded estimate; after the trigger it returns a spike
        re-encoding the frozen value and ignores ``input``.

        Args:
            input: Tensor of current 0/1 spikes in the configured polarity.

        Returns:
            0/1 output spike tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        # A progressive decode of the live stream tracks the reusable estimate on every timestep.
        self.decoder(input)
        if self.timestep_cur < self.trigger_timestep:
            # Before the trigger, the live input stream passes through unchanged.
            return input
        if not self.latched:
            # At the trigger, freeze the running decoded estimate over the first trigger_timestep samples.
            estimate = self.decoder.spike_value.detach()
            if self.held.shape == estimate.shape:
                self.held.copy_(estimate)
            else:
                self.held.resize_as_(estimate).copy_(estimate)
            self.latched = True
        # After the trigger, re-emit a fresh spike stream of the frozen value.
        reference_encode_bit = self.reference_encode(self.held)
        return reference_encode_bit

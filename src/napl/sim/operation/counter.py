import torch
from loguru import logger

from napl.sim.base import napl_base


class counter(napl_base):
    r"""
    Count input spikes into a bounded running register and read out its ceiling.

    Use this operation as a streaming "value variable": each timestep it adds the
    current input spikes to a running count and holds that count in a register.
    The count is bounded by a **width**-bit ceiling ``max_count = 2**width - 1``:
    once the running count reaches ``max_count`` it saturates there and stays,
    and any further input spikes are dropped rather than wrapping the register.

    The per-timestep readout is the saturation flag of that bounded count: it is
    ``1`` from the timestep the running count first reaches ``max_count`` onward,
    and ``0`` before then. Decoded over a stream, the output rate is the fraction
    of the stream that follows the arrival of the ``max_count``-th input spike, so
    the operation reports, as a unipolar spike stream, that the accumulator has
    counted at least ``max_count`` spikes so far.

    A running spike count is non-negative, so only unipolar ``0``/``1`` input is
    supported; there is no signed-count (bipolar) form.

    Call :meth:`clear_count` to zero the running count mid-stream without a full
    :meth:`reset`, which also rewinds ``timestep_cur``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import counter

        acc = counter({'width': 3})
        output = acc(torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        Saturating running-count accumulator derived from the counter-state
        readout of *uGEMM: Unary Computing Architecture for GEMM Applications*,
        ISCA, 2020.
    """


    def __init__(
            self,
            config={'width': 3}
        ):
        """
        Configure the counter ceiling.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Counter bit width; the running count saturates at
                ``2**width - 1``. The default is ``3``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['width'], optional_key_list=['polarity'], polarity_required=False)
        if config['width'] < 1:
            message = f'Counter width <{config["width"]}> must be at least 1.'
            logger.error(message)
            raise AssertionError(message)

        #: Counter bit width.
        self.width = config['width']
        #: Largest value the running count saturates to.
        self.max_count = 2 ** self.width - 1
        #: Saturating running spike count, allocated as a scalar and broadcast to the input shape on first use.
        self.count: torch.Tensor
        self.register_buffer('count', torch.zeros(1, dtype=self.ntype))
        # The saturation flag is combinational on the just-updated count, so input reaches output the same cycle.
        #: Hardware latency and timing metadata, with zero input-to-output delay.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'unipolar', 'output': 'unipolar'}
        self.correlation_i = {}


    def _reset(self):
        """
        Clear the running count and shrink it back to a scalar seed.

        The next call reallocates the count to the input shape, so a replay after
        :meth:`reset` starts from an empty accumulator.
        """
        self.count.resize_(1).zero_()


    def forward(self, input: torch.Tensor):
        """
        Add one timestep of input spikes to the running count and read its ceiling.

        Args:
            input: Unipolar ``0``/``1`` spike tensor for the current timestep.

        Returns:
            A ``0``/``1`` spike tensor, ``1`` once the running count has reached
            ``max_count`` and ``0`` before then, with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = acc(torch.tensor([1], dtype=torch.int8))
        """
        delta = input.type(self.ntype)
        # Accumulate this timestep and saturate; the scalar seed broadcasts to the input shape once.
        if self.count.shape == delta.shape:
            self.count.add_(delta).clamp_(0, self.max_count)
        else:
            updated = self.count.add(delta).clamp_(0, self.max_count)
            self.count.resize_as_(updated).copy_(updated.detach())
        return self.count.ge(self.max_count).type(self.stype)


    def clear_count(self):
        """
        Zero the running count mid-stream, keeping the current timestep position.

        Unlike :meth:`reset`, this leaves ``timestep_cur`` and the count shape
        untouched, so a caller can restart the accumulation without ending the run.
        The next call counts up from an empty accumulator again.
        """
        self.count.zero_()

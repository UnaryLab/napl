import torch

from napl.sim.base import napl_base, hw_params
from loguru import logger


class add_any(napl_base):
    """
    Add spike streams with a configurable output scale.

    Use this streaming accumulator when the reduced sum must emit one output
    spike per ``scale`` accumulated units. It supports unipolar and bipolar
    rate-coded inputs.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import add_any

        adder = add_any({'polarity': 'unipolar', 'scale': 2, 'width': 10})
        output = adder(torch.tensor([[1, 0], [1, 1]], dtype=torch.int8), dim=0)
    """
    def __init__(
            self,
            config={
                'polarity' : 'bipolar',
                'scale' : 2,
                'width' : 10,
            }
        ):
        """
        Configure the carry scale and accumulator width.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, either ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **scale**: Accumulated amount required for one output spike; the default is ``2``.
              - **width**: Signed accumulator width in bits; the default is ``10``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'scale', 'width'], polarity_required=True)
        self.hw = hw_params(pp_delay=0)

        # width of the accumulator
        self.width = config['width']
        # max value in the accumulator
        self.acc_max = 2**(self.width-1) - 1
        # min value in the accumulator
        self.acc_min = -2**(self.width-1)

        # the carry scale at the output
        self.register_buffer('scale', torch.tensor(config['scale'], dtype=self.ntype))
        # accumulation offset
        self.offset = 0
        # accumulator for (PC - offset)
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
        self.is_first_call = True


    def _reset(self):
        """
        Clear the local accumulator and first-call shape state.
        """
        self.accumulator.resize_(1).zero_()
        self.is_first_call = True


    def forward(self, input, entry=None, dim=-1):
        """
        Accumulate and emit one output timestep.

        Args:
            input: Current spike tensor, or a pre-reduced partial sum when
                ``dim=None``.
            entry: Number of source streams used for the bipolar offset. It is
                inferred from ``input.size(dim)`` unless ``dim=None``.
            dim: Dimension to reduce. Set it to ``None`` for pre-reduced input;
                the default is ``-1``.

        Returns:
            A spike tensor with the reduced dimension removed. The call updates
            the accumulator and advances the module by one timestep.

        **Example:**

        .. code-block:: python

            output = adder(torch.tensor([1, 1], dtype=torch.int8), dim=0)
        """
        if self.is_first_call:
            if self.polarity == 'bipolar':
                if entry is None:
                    assert dim is not None, \
                        logger.error('add_any with pre-reduced input (dim=None) requires an explicit <entry>.')
                    entry = input.size()[dim]
                self.offset = (entry - self.scale)/2

            self.is_first_call = False

        if dim is None:
            acc_delta = input.type(self.ntype) - self.offset
        else:
            # in-place sub on the freshly-allocated sum (owned temp): same
            # (partial - offset) math as before, one fewer full-size alloc per timestep
            acc_delta = torch.sum(input, dim, dtype=self.ntype)
            acc_delta.sub_(self.offset)
        # in-place add/clamp once the accumulator matches the stream shape (both ntype,
        # so promotion is a no-op); the first timestep must broadcast-expand the (1,)
        # init, which add_ cannot do
        if self.accumulator.shape == acc_delta.shape:
            self.accumulator.add_(acc_delta).clamp_(self.acc_min, self.acc_max)
        else:
            updated = self.accumulator.add(acc_delta).clamp(self.acc_min, self.acc_max)
            self.accumulator.resize_as_(updated).copy_(updated.detach())
        output = torch.ge(self.accumulator, self.scale).type(self.ntype)
        # subtract scale only where output==1 (acc>=scale>0), fused, no intermediate
        # alloc; result stays in [0, acc_max] so the post-clamp would be a no-op
        self.accumulator.addcmul_(output, self.scale, value=-1)
        return output.type(self.stype)

import torch

from napl.sim.base import napl_base, hw_params
from loguru import logger


class add_any(napl_base):
    """
    Add spike streams with a configurable output scale.

    Use this streaming accumulator when the reduced sum must emit one output
    spike per ``scale`` accumulated units. It supports unipolar and bipolar
    rate-coded inputs.

    The precise target reductions are

    .. math::

       p_y = \\frac{1}{\\mathit{scale}}\\sum_i p_i
       \\quad (\\text{unipolar}),\\qquad
       v_y = \\frac{1}{\\mathit{scale}}\\sum_i v_i
       \\quad (\\text{bipolar}).

    Let ``r_t`` be ``input`` when ``dim=None`` and otherwise the sum of
    ``input`` along ``dim``. With ``o = 0`` for unipolar input and
    ``o = (entry - scale) / 2`` for bipolar input, the exact carry recurrence is

    .. math::

       \\tilde a_t = \\operatorname{clip}(a_{t-1} + r_t - o,
       -2^{w-1}, 2^{w-1}-1),\\qquad
       y_t = \\mathbf{1}\\{\\tilde a_t \\geq \\mathit{scale}\\},\\qquad
       a_t = \\tilde a_t - \\mathit{scale} y_t.

    The output removes the reduced dimension and emits one spike for each
    inclusive ``scale`` threshold crossing.

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

        #: Signed accumulator width in bits.
        self.width = config['width']
        #: Largest value retained by the signed accumulator.
        self.acc_max = 2**(self.width-1) - 1
        #: Smallest value retained by the signed accumulator.
        self.acc_min = -2**(self.width-1)
        assert config['scale'] <= self.acc_max, logger.error(
            f'add_any scale <{config["scale"]}> exceeds accumulator maximum '
            f'<{self.acc_max}> for width <{self.width}>.'
        )

        #: Accumulated amount consumed when an output spike is emitted.
        self.scale = config['scale']
        #: Bipolar centering offset inferred from the input count on first use.
        self.offset = 0
        #: Running centered input sum used to decide when to emit a spike.
        self.accumulator: torch.Tensor
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
        #: Whether the next call must infer input-dependent state.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the combinational adder.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


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
            acc_delta = torch.sum(input, dim, dtype=self.ntype)
            acc_delta.sub_(self.offset)
        # The scalar initial state broadcasts out of place; matching shapes update in place.
        if self.accumulator.shape == acc_delta.shape:
            self.accumulator.add_(acc_delta).clamp_(self.acc_min, self.acc_max)
        else:
            updated = self.accumulator.add(acc_delta).clamp(self.acc_min, self.acc_max)
            self.accumulator.resize_as_(updated).copy_(updated.detach())
        output = torch.ge(self.accumulator, self.scale).type(self.ntype)
        # With scale > 0, emitting a carry preserves the accumulator bounds.
        self.accumulator.sub_(output, alpha=self.scale)
        return output.type(self.stype)

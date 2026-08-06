import torch

from napl.sim.base import napl_base, hw_params


class avgpool2d(napl_base):
    r"""Average-pool a unary spike stream one timestep at a time.

    Use this module as the unary counterpart of ``torch.nn.AvgPool2d``. The
    emitted spike rate represents the pooling window mean for either unipolar or
    bipolar encoding,

    .. math::

       y = \mathrm{avgpool2d}(x),

    with a residual accumulator error below one spike.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import avgpool2d

        pool = avgpool2d(2, config={"polarity": "unipolar"})
        output_spike = pool(torch.ones(1, 1, 2, 2))
        assert output_spike.shape == (1, 1, 1, 1)
    """


    def __init__(self, kernel_size, stride=None, config={'polarity': 'bipolar'}):
        """Configure the pooling geometry and unary representation.

        .. container:: api-parameter-list

            **Parameters:**

            - **kernel_size** – Pooling window size accepted by ``torch.nn.AvgPool2d``.
            - **stride** – Pooling stride, where ``None`` uses ``kernel_size``; the default is ``None``.
            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **name**: Optional instance label.

        Construction initializes a scalar accumulator that expands to the pooled
        output shape on the first call.
        """
        super().__init__(config, ['polarity'], polarity_required=True)
        #: PyTorch pooling operator that computes each per-timestep window mean.
        self.avgpool2d = torch.nn.AvgPool2d(kernel_size, stride=stride)
        #: Residual pooled value carried forward until it emits an output spike.
        self.accumulator: torch.Tensor
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))

        # Pooling and the accumulator threshold are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming pool.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input_spike': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_spike': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Clear the local pooling accumulator.

        The accumulator returns to a scalar zero and will adopt the next output
        shape on demand. This hook returns ``None`` and is called by ``reset()``.
        """
        self.accumulator.resize_(1).zero_()


    def forward(self, input_spike):
        """Process one spatial spike tensor.

        Args:
            input_spike: Current ``(batch, channel, height, width)`` spike tensor.

        Returns:
            A spike tensor with the shape produced by ``AvgPool2d``.

        The call updates the persistent accumulator. Calling the module also
        advances ``timestep_cur`` once.
        """
        pooled_input = input_spike if input_spike.dtype == self.ntype else input_spike.type(self.ntype)
        delta = self.avgpool2d(pooled_input)
        if self.accumulator.shape == delta.shape:
            self.accumulator.add_(delta)
        else:
            # The first update broadcasts the scalar state out of place.
            expanded = self.accumulator.add(delta).detach()
            self.accumulator.resize_as_(expanded).copy_(expanded)
        # sub_ promotes 0/1 spikes into the accumulator dtype without changing values.
        output = torch.ge(self.accumulator, 1).type(self.stype)
        self.accumulator.sub_(output)
        return output

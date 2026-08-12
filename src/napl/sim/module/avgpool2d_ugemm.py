import torch

from torch.nn.modules.utils import _pair

from napl.sim.base import napl_base
from napl.sim.operation import add_scale
from loguru import logger


class avgpool2d_ugemm(napl_base):
    r"""Average-pool a unary spike stream one timestep at a time.

    Use this module as the unary counterpart of ``torch.nn.AvgPool2d``. The
    emitted spike rate represents the pooling window mean for either unipolar or
    bipolar encoding,

    .. math::

       y = \mathrm{avgpool2d\_ugemm}(x).

    The window mean is exact for every kernel size, up to the residual held in
    the accumulator, which stays below one spike.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.module import avgpool2d_ugemm

        pool = avgpool2d_ugemm(2, config={"polarity": "unipolar"})
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
              - **width**: Signed accumulator width of the internal adder, which must satisfy ``2 ** (width - 1) - 1 >= (scale - grid) + delta_max``; with the unipolar child's ``scale = delta_max = kernel_area`` on an integer grid (``grid = 1``) this reduces to ``2 ** (width - 1) - 1 >= 2 * kernel_area - 1``; the default is ``12``. It is simulation-only: the RTL derives its own accumulator width as ``clog2(4 * KERNEL_AREA + 2)`` and ``src/napl/imp/mapping.yaml`` exposes only ``KERNEL_AREA`` and ``LANES``, so ``width`` is not plumbed to hardware.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity'], optional_key_list=['width'], polarity_required=True)
        #: PyTorch pooling operator that counts the spikes in each window.
        self.avgpool2d = torch.nn.AvgPool2d(kernel_size, stride=stride, divisor_override=1)
        kh, kw = _pair(kernel_size)
        #: Number of spikes covered by one pooling window.
        self.kernel_area = kh * kw

        width = config.get('width', 12)
        if not isinstance(width, int):
            message = f'avgpool2d_ugemm accumulator width must be int: got <{width}>.'
            logger.error(message)
            raise AssertionError(message)
        # Before thresholding the accumulator holds the largest sub-threshold residue
        # (scale - grid) plus one timestep's step, which is kernel_area - 1 plus
        # kernel_area for the unipolar child on an integer grid.
        if 2 ** (width - 1) - 1 < (self.kernel_area - 1) + self.kernel_area:
            message = (
                f'avgpool2d_ugemm accumulator width <{width}> too small for kernel area '
                f'<{self.kernel_area}>: 2**(width-1) - 1 must be >= 2*kernel_area - 1 or '
                f'partial sums saturate. Increase width.'
            )
            logger.error(message)
            raise AssertionError(message)

        # The window mean of the input rates is the target under both polarities, and the
        # bipolar offset (entry - scale) / 2 vanishes at scale = entry, so one unipolar
        # adder at scale = entry = kernel_area serves both.
        #: Streaming unary adder that averages each window spike count.
        self.acc = add_scale({'polarity': 'unipolar', 'scale': self.kernel_area,
                              'intwidth': width, 'fracwidth': 0})

        # Pooling and the adder threshold are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming pool.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset state owned directly by the pool.

        This class has no extra local state. The inherited ``reset()`` method
        resets the registered unary adder.
        """
        pass


    def forward(self, input):
        """Process one spatial spike tensor.

        Args:
            input: Current ``(batch, channel, height, width)`` spike tensor.

        Returns:
            A spike tensor with the shape produced by ``AvgPool2d``.

        The call passes the window spike count to the unary adder, which emits one
        spike per ``kernel_area`` accumulated spikes. Calling the module also
        advances ``timestep_cur`` once.
        """
        pooled_input = input if input.dtype == self.ntype else input.type(self.ntype)
        popcount = self.avgpool2d(pooled_input)
        return self.acc(popcount, entry=self.kernel_area, dim=None)

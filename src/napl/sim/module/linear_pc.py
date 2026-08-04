import torch
import math

from napl.sim.base import napl_base
from loguru import logger


class linear_pc(napl_base):
    r"""Return the per-timestep parallel count of a unary linear product.

    Use this streaming layer when downstream logic needs the raw product count
    rather than a scaled output bitstream. It returns the per-timestep binary inner-product
    count of the input spikes against freshly encoded weight spikes, before any accumulation
    into a bitstream. This is the :class:`linear` partial sum without its scaled
    unary adder.

    Each timestep the weights (and bias) are encoded into spikes on a distinct RNG dimension
    from the input (decorrelated operands). For unipolar this returns the AND-count
    ``sum(input & weight)`` plus the bias spike; for bipolar it returns the
    XNOR count ``sum(input == weight)`` plus the bias spike on the input-``1``
    path, matching ``FSULinearPC``. The count per timestep lies in
    ``[0, entry]``, where ``entry = in_features + has_bias``. Accumulating the
    count over ``T`` timesteps and dividing by ``T`` recovers the unipolar inner product directly,
    or the bipolar inner product as ``2 * mean - entry``. This class matches
    UnarySim ``FSULinearPC``.

    The precise target is the product count whose time average recovers the
    inner product,

    .. math::

       \frac{1}{T}\sum_{t=1}^{T} c_t = Wx + b \ \ (\text{unipolar}),\qquad
       \frac{2}{T}\sum_{t=1}^{T} c_t - e = Wx + b \ \ (\text{bipolar}),

    with :math:`e = n + [\,\text{bias}\,]`. Each timestep the layer returns that
    count exactly, with :math:`w_t` the freshly encoded weight spikes and
    :math:`b_t` the bias spike,

    .. math::

       c_t = \begin{cases}
       x_t w_t^{\top} + b_t, & \text{unipolar},\\
       2 x_t w_t^{\top} - \sum_j x_{j,t} - \sum_j w_{j,t} + n + b_t,
       & \text{bipolar},
       \end{cases}

    so :math:`c_t \in [0, e]` and no accumulation is applied.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_pc

        counter = linear_pc(torch.ones(3, 2),
                            config={"polarity": "unipolar", "timestep": 4,
                                    "generator": "sobol"})
        count = counter(torch.ones(1, 2))

    References
    ----------
    *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """


    def __init__(
            self,
            weight,
            bias=None,
            config={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
                'dim': 2,
            }
        ):
        """Construct the counter from external numeric weights and bias.

        Args:
            weight: Numeric tensor shaped ``(out_features, in_features)``.
            bias: Optional numeric tensor shaped ``(out_features,)``. Defaults
                to ``None``.
            config: Configuration mapping with **polarity** (default
                ``"bipolar"``), **timestep** (default ``256``), **generator**
                (default ``"sobol"``), and **dim** (weight Sobol dimension,
                default ``2``; bias uses ``dim + 1``). **name** is an optional
                instance label and defaults to ``None``.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # This import stays local to avoid the module-operation import cycle.
        from napl.sim.operation.encode import encode

        assert weight.dim() == 2, logger.error(f'linear_pc weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
        #: Trainable numeric weight encoded into a spike stream.
        self.weight = torch.nn.Parameter(weight)
        #: Optional trainable numeric bias encoded on its own sequence.
        self.bias = torch.nn.Parameter(bias) if bias is not None else None
        #: Number of features produced by the counter.
        self.out_features = weight.shape[0]
        #: Number of features consumed by the counter.
        self.in_features = weight.shape[1]
        #: Whether an encoded bias contributes to each output count.
        self.has_bias = bias is not None
        #: Parallel-count fan-in, including the bias when present.
        self.entry = self.in_features + (1 if self.has_bias else 0)

        dim = config.get('dim', 2)
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'linear_pc decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim (identical sequences across '
                f'operands). Use a sobol-family generator, or decorrelate the input and weight '
                f'streams by distinct seeds.')
        #: Encoder that converts the numeric weight to spikes.
        self.w_encoder = encode({'polarity': self.polarity, 'timestep': config['timestep'],
                                  'generator': config['generator'], 'dim': dim})
        if self.has_bias:
            #: Encoder that converts the optional numeric bias to spikes.
            self.b_encoder = encode({'polarity': self.polarity, 'timestep': config['timestep'],
                                      'generator': config['generator'], 'dim': dim + 1})

        self.encoding_io = {'input_spike': 'rc'}
        self.polarity_io = {'input_spike': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset state owned directly by the counter.

        This class has no extra local state. The inherited ``reset()`` method
        resets its registered encoders.
        """
        pass


    def forward(self, input_spike):
        """Count spike products for one timestep.

        Args:
            input_spike: ``0``/``1`` tensor whose last dimension is
                ``in_features``.

        Returns:
            Numeric count tensor with last dimension ``out_features``. Each
            element is in ``[0, in_features + has_bias]``.

        The call advances the counter and its encoders. It does not accumulate
        counts across timesteps.
        """
        w_spike = self.w_encoder(self.weight)
        xf = input_spike.type(self.ntype)
        wf = w_spike.type(self.ntype)
        and_count = torch.matmul(xf, wf.t())
        pc = and_count
        if self.has_bias:
            # Bias contributes only to the input-one path.
            pc = pc + self.b_encoder(self.bias).type(self.ntype)
        if self.polarity == 'bipolar':
            # sum((1-x)(1-w)) = in_features - sum(x) - sum(w) + sum(xw).
            input0 = self.in_features - xf.sum(-1, keepdim=True) - wf.sum(-1) + and_count
            pc = pc + input0
        return pc

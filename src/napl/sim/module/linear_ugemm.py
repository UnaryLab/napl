import torch

from napl.sim.base import napl_base
from napl.sim.operation import add_any, mul_ugemm
from napl.sim.operation import encode
from napl.sim.module._shared import _check_acc_width
from loguru import logger


class linear_ugemm(napl_base):
    r"""Apply a streaming unary linear layer with conditional spike generation.

    Use this layer when input-driven uGEMM weight streams are preferred over the
    free-running weight encoder used by :class:`linear_mix`. Each input feature
    advances its number-sequence index only when its input spike is ``1``, and
    bipolar mode adds an input-``0`` path on a separate index, so the layer
    computes the scaled affine map one timestep at a time,

    .. math::

       y = \frac{Wx + b}{s},

    with **scale** :math:`s` defaulting to ``entry = in_features + has_bias``.
    The output rate reaches that target within the stochastic-computing error of
    the conditionally generated streams. Only UnarySim's
    ``FSULinearuGEMM(scaled=True)`` accumulation is implemented; the non-scaled
    output-comparator variant is not. This class matches UnarySim
    ``FSULinearuGEMM`` in scaled mode.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_ugemm

        layer = linear_ugemm(torch.zeros(3, 2),
                             config={"polarity": "bipolar", "timestep": 4,
                                     "generator": "sobol"})
        output_spike = layer(torch.ones(1, 2))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """
    #: Encoding advances conditionally on data, so the RTL counterpart holds
    #: its own encoder instead of sharing an external one.
    internal_encode = True


    def __init__(
            self,
            weight,
            bias=None,
            config={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
                'dim': 1,
                'scale': None,
                'width': 8,
            }
        ):
        """Construct the streaming CSG layer from external numeric parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **weight** – Numeric tensor shaped ``(out_features, in_features)``.
            - **bias** – Optional numeric tensor shaped ``(out_features,)``; the default is ``None``.
            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **timestep**: Positive stream length; the default is ``256``.
              - **generator**: Number-sequence generator name; the default is ``"sobol"``.
              - **dim**: Number-sequence dimension; the default is ``1``.
              - **scale**: Output divisor, where ``None`` uses ``in_features + has_bias``; the default is ``None``.
              - **width**: Signed accumulator width, which must satisfy ``2 ** (width - 1) - 1 >= (scale - grid) + delta_max``, where ``delta_max`` is the largest per-timestep accumulator step (``entry`` when unipolar, ``(entry + scale) / 2`` when bipolar, with ``entry = in_features + has_bias``) and ``grid`` is the accumulator step (``0.5`` when bipolar with odd ``entry - scale``, else ``1``); the default is ``8``. This bound is static for ``scale >= entry``; for ``scale < entry`` the width must also satisfy ``2 ** (width - 1) > entry``, a minimum burst-headroom floor rather than a safety bound, since the accumulator then drains by at most ``scale`` per timestep and correctness is conditional on the long-run mean inflow staying below ``scale`` (see :class:`add_any`).
              - **name**: Optional instance label.

        Weight and bias are updatable only by an in-place write, such as
        ``with torch.no_grad(): layer.weight.fill_(1.0)``. Each timestep reads
        their current values, so the write takes effect on the next call,
        without a ``reset()``.

        .. warning::

            Optimizers silently skip these ``Parameter`` objects. The spike
            comparison is not differentiable, so no gradient ever reaches them,
            ``.grad`` stays ``None``, and ``SGD.step()`` leaves the values
            bit-identical.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['dim', 'scale', 'width'], polarity_required=True)

        if weight.dim() != 2:
            message = f'linear_ugemm weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.'
            logger.error(message)
            raise AssertionError(message)
        #: Number of output features produced by the layer.
        self.out_features = weight.shape[0]
        #: Number of input features consumed by the layer.
        self.in_features = weight.shape[1]
        #: Whether an encoded bias contributes to the parallel count.
        self.has_bias = bias is not None
        #: Parallel-count fan-in, including the bias when present.
        self.entry = self.in_features + (1 if self.has_bias else 0)
        scale = config.get('scale', None)
        #: Divisor implemented by the streaming unary adder.
        self.scale = self.entry if scale is None else scale

        #: Requested number of output-spike timesteps in the stream.
        self.timestep = config['timestep']
        if self.timestep <= 0:
            message = f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.'
            logger.error(message)
            raise AssertionError(message)

        #: Signed accumulator width used by the streaming unary adder.
        self.width = _check_acc_width('linear_ugemm', config.get('width', 8), self.entry,
                                      self.scale, self.polarity)

        self._is_bipolar = (self.polarity == 'bipolar')
        #: Externally updatable numeric weight matrix converted to spike probabilities on use.
        self.weight = torch.nn.Parameter(weight)
        #: Optional externally updatable numeric bias converted to spike probabilities on use.
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

        # Weight and bias bits share one RNG; input-driven indices decorrelate products.
        #: Conditional spike generator holding the weight sequence and its per-input indices.
        self.mul = mul_ugemm({'polarity': self.polarity, 'timestep': self.timestep,
                              'generator': config['generator']})
        # mul_ugemm builds its sequence on dimension 1; this layer selects its own.
        self.mul.num_seq.copy_(encode({'polarity': self.polarity,
                                       'timestep': self.mul.len,
                                       'generator': config['generator'],
                                       'dim': config.get('dim', 1)}).num_seq)

        #: Streaming unary adder that reduces each linear product count.
        self.acc = add_any({'polarity': self.polarity, 'scale': self.scale, 'width': self.width})

        # Conditional generation and the adder are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming layer.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_spike': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_spike': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset state owned directly by the layer.

        This class has no extra local state. The inherited ``reset()`` method
        restarts the conditional spike generator, which owns the per-input
        sequence indices, and the unary adder.
        """
        pass


    def forward(self, input_spike):
        """Process one input-spike timestep.

        Args:
            input_spike: ``0``/``1`` tensor whose last dimension is
                ``in_features``.

        Returns:
            Output spike tensor with last dimension ``out_features``.

        The call advances the per-feature conditional RNG indices, the unary
        adder, and ``timestep_cur``. Weight and bias spike probabilities are
        recomputed from the current parameters on every call.
        """
        # The input spike broadcasts over output features, so each input feature
        # keeps one sequence index shared by the whole weight column.
        # Counts bounded by entry are exact float32 integers under any reduction order.
        product = self.mul(input_spike.unsqueeze(-2), self.weight)
        psum = product.type(self.ntype).sum(-1)

        if self.has_bias:
            # Bias advances once per timestep; bool promotion preserves its 0/1 value.
            b_prob = ((self.bias + 1) / 2 if self._is_bipolar else self.bias).type(self.ntype)
            b_bit = torch.gt(b_prob, self.mul.num_seq[(self.timestep_cur - 1) % self.mul.len])
            psum = psum + b_bit

        return self.acc(psum, entry=self.entry, dim=None)

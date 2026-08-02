import torch
import math

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger


class linear_ugemm(napl_base):
    """Apply a streaming unary linear layer with conditional spike generation.

    Use this layer when input-driven uGEMM weight streams are preferred over the
    free-running weight encoder used by :class:`linear`. It computes
    ``y = W x (+ b)`` bit by bit. Each input feature advances its RNG index only
    when its input spike is ``1``; bipolar mode adds an input-``0`` path with a
    separate index. The per-timestep products are summed by ``add_any`` and the
    decoded output represents ``(W x + b) / scale``, with **scale** defaulting
    to ``entry = in_features + has_bias``. Only UnarySim's
    ``FSULinearuGEMM(scaled=True)`` accumulation is implemented; the
    non-scaled output-comparator variant is not. This class matches UnarySim
    ``FSULinearuGEMM`` in scaled mode.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_ugemm

        layer = linear_ugemm(torch.zeros(3, 2),
                             config={"polarity": "bipolar", "timestep": 4,
                                     "generator": "sobol"})
        output_spike = layer(torch.ones(1, 2))

    References
    ----------
    *uGEMM: Unary Computing Architecture for GEMM Applications*.
    """


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
                'width': 12,
            }
        ):
        """Construct the streaming CSG layer from external numeric parameters.

        Args:
            weight: Numeric tensor shaped ``(out_features, in_features)``.
            bias: Optional numeric tensor shaped ``(out_features,)``. Defaults to
                ``None``.
            config: Configuration mapping with these keys:

                * **polarity** - ``"unipolar"`` or ``"bipolar"``. Defaults to
                  ``"bipolar"``.
                * **timestep** - Positive stream length. Defaults to ``256``.
                * **generator** - Number-sequence generator. Defaults to
                  ``"sobol"``.
                * **dim** - Number-sequence dimension. Defaults to ``1``.
                * **scale** - Output divisor. ``None`` uses
                  ``in_features + has_bias``. Defaults to ``None``.
                * **width** - Signed accumulator width. Defaults to ``12`` and
                  must satisfy ``2 ** (width - 1) >= in_features + has_bias``.
                * **name** - Optional instance label. Defaults to ``None``.

        Weight and bias are trainable parameters. The layer converts their
        current values to spike probabilities at each timestep.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # These imports stay local to avoid the module-operation import cycle.
        from napl.sim.operation import add_any
        from napl.sim.module.encoder import gen_num_seq

        assert weight.dim() == 2, logger.error(
            f'linear_ugemm weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
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
        assert self.timestep > 0, logger.error(
            f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        #: Bit width of the power-of-two number-sequence period.
        self.seq_width = math.ceil(math.log2(self.timestep))
        #: Number of thresholds in the periodic number sequence.
        self.len = 2 ** self.seq_width

        # The signed accumulator range must contain every per-step partial sum.
        width = config.get('width', 12)
        assert 2 ** (width - 1) >= self.entry, logger.error(
            f'linear_ugemm accumulator width <{width}> too small for fan-in <{self.entry}>: '
            f'2**(width-1) must be >= entry or partial sums saturate. Increase width.')

        self._is_bipolar = (self.polarity == 'bipolar')
        #: Trainable numeric weight matrix converted to spike probabilities on use.
        self.weight = torch.nn.Parameter(weight)
        #: Optional trainable numeric bias converted to spike probabilities on use.
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

        # Weight and bias bits share one RNG; input-driven indices decorrelate products.
        #: Threshold sequence shared by weight and bias spike generation.
        self.num_seq: torch.Tensor
        self.register_buffer(
            'num_seq',
            gen_num_seq({'width': self.seq_width,
                         'generator': config['generator'],
                         'dim': config.get('dim', 1)}),
        )

        # Each input feature advances its own input-one RNG index.
        #: Per-input conditional-generator indices advanced by input-one spikes.
        self.seq_idx: torch.Tensor
        self.register_buffer('seq_idx', torch.zeros(1, dtype=torch.long))
        if self._is_bipolar:
            #: Per-input indices advanced by input-zero spikes in bipolar mode.
            self.seq_idx_inv: torch.Tensor
            self.register_buffer('seq_idx_inv', torch.zeros(1, dtype=torch.long))

        #: Streaming unary adder that reduces each linear product count.
        self.acc = add_any({'polarity': self.polarity, 'scale': self.scale, 'width': width})


    def _reset(self):
        """Reset the local conditional-generator indices.

        The input-one path index and, for bipolar streams, the input-zero path
        index return to scalar zero. The unary adder is reset by ``reset()`` as a
        registered child.
        """
        self.seq_idx.resize_(1).zero_()
        if self._is_bipolar:
            self.seq_idx_inv.resize_(1).zero_()


    def forward(self, input_spike):
        """Process one input-spike timestep.

        Args:
            input_spike: ``0``/``1`` tensor whose last dimension is
                ``in_features``.

        Returns:
            Output spike tensor with last dimension ``out_features``.

        The call advances the per-feature conditional RNG indices, the unary
        adder, and ``timestep_cur``. Stored spike probabilities are unchanged.
        """
        xf = input_spike.type(self.ntype)
        x_long = input_spike.type(torch.long)
        w_prob = ((self.weight + 1) / 2 if self._is_bipolar else self.weight).type(self.ntype)

        # Counts bounded by entry are exact float32 integers under any reduction order.
        thr = self.num_seq[self.seq_idx]
        w_bit = torch.gt(w_prob, thr.unsqueeze(-2)).type(self.ntype)
        psum = torch.matmul(w_bit, xf.unsqueeze(-1)).squeeze(-1)
        if self.seq_idx.shape == x_long.shape:
            # Matching long state updates in place without aliasing another timestep.
            self.seq_idx.add_(x_long)
        else:
            expanded = self.seq_idx.add(x_long).detach()
            self.seq_idx.resize_as_(expanded).copy_(expanded)

        if self.has_bias:
            # Bias advances once per timestep; bool promotion preserves its 0/1 value.
            b_prob = ((self.bias + 1) / 2 if self._is_bipolar else self.bias).type(self.ntype)
            b_bit = torch.gt(b_prob, self.num_seq[(self.timestep_cur - 1) % self.len])
            psum = psum + b_bit

        if self._is_bipolar:
            thr_inv = self.num_seq[self.seq_idx_inv]
            w_bit_inv = torch.le(w_prob, thr_inv.unsqueeze(-2)).type(self.ntype)
            psum = psum + torch.matmul(w_bit_inv, (1 - xf).unsqueeze(-1)).squeeze(-1)
            if self.seq_idx_inv.shape == x_long.shape:
                self.seq_idx_inv.add_(1 - x_long)
            else:
                expanded = self.seq_idx_inv.add(1 - x_long).detach()
                self.seq_idx_inv.resize_as_(expanded).copy_(expanded)

        return self.acc(psum, entry=self.entry, dim=None)

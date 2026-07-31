import torch
import math

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger


class linear_ugemm(napl_base):
    """Apply a streaming unary linear layer with conditional spike generation.

    Use this layer when input-driven uGEMM weight streams are preferred over the
    free-running weight encoder used by :class:`linear`. It provides conditional spike
    generation: y = W x (+ b), computed bit by bit from *binary* weights. Unlike
    linear (which encodes the weights on an independent RNG each timestep), the
    weight spikes are generated conditionally on the input spikes, mul_csg style: each
    input feature advances its RNG index only when its input spike is 1 (bipolar adds
    the complementary input-0 path with its own index), so the weight bit generation is
    input-driven and needs no separate weight encoder. The per-timestep partial products
    are summed by the scaled unary adder (add_any); the decoded output represents
    (W x + b) / scale with scale defaulting to entry = in_features + has_bias.
    Only UnarySim's scaled accumulation is ported (FSULinearuGEMM(scaled=True)); the
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

        Numeric parameters are converted to spike probabilities. They are not
        registered as trainable parameters by this class.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # lazy import: operation.mul_csg imports module.encoder, so importing operation at
        # module top would create an import cycle with module/__init__.
        from napl.sim.operation import add_any
        from napl.sim.module.encoder import gen_num_seq

        assert weight.dim() == 2, logger.error(
            f'linear_ugemm weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
        self.out_features, self.in_features = weight.shape
        self.has_bias = bias is not None
        self.entry = self.in_features + (1 if self.has_bias else 0)
        scale = config.get('scale', None)
        self.scale = self.entry if scale is None else scale

        self.timestep = config['timestep']
        assert self.timestep > 0, logger.error(
            f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        self.seq_width = math.ceil(math.log2(self.timestep))
        self.len = 2 ** self.seq_width

        # the scaled accumulator must hold a per-step partial sum up to `entry`; if the
        # accumulator range 2**(width-1) is smaller it saturates and silently returns
        # near-maximal error, so reject that configuration outright.
        width = config.get('width', 12)
        assert 2 ** (width - 1) >= self.entry, logger.error(
            f'linear_ugemm accumulator width <{width}> too small for fan-in <{self.entry}>: '
            f'2**(width-1) must be >= entry or partial sums saturate. Increase width.')

        self._is_bipolar = (self.polarity == 'bipolar')
        # binary weight/bias held as compare probabilities (rate of the target stream)
        self.w_prob = ((weight + 1) / 2 if self._is_bipolar else weight).type(self.ntype)
        if self.has_bias:
            self.b_prob = ((bias + 1) / 2 if self._is_bipolar else bias).type(self.ntype)

        # one shared RNG sequence for weight and bias bit generation (UnarySim uses a
        # single Sobol dim-1 RNG for both); the weight indices are input-driven so the
        # decorrelation-by-dim concern of linear does not apply here.
        self.register_buffer(
            'num_seq',
            gen_num_seq({'width': self.seq_width,
                         'generator': config['generator'],
                         'dim': config.get('dim', 1)}),
        )

        # per-input-feature RNG index, advanced by the input spike (input-1 path);
        # scalar init broadcasts up to the input shape on the first forward().
        self.register_buffer('seq_idx', torch.zeros(1, dtype=torch.long))
        if self._is_bipolar:
            # input-0 path index, advanced by the complemented input spike
            self.register_buffer('seq_idx_inv', torch.zeros(1, dtype=torch.long))

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
        # input_spike: (..., in_features) spike tensor for the current timestep
        xf = input_spike.type(self.ntype)
        x_long = input_spike.type(torch.long)

        # input-1 path: weight bit per (out, in) from the input-driven RNG index; the
        # 0/1 mask-and-reduce is a matmul (one fused kernel, no (out, in) mul temp;
        # bit-exact: 0/1 sums bounded by entry are exact float32 integers in any order)
        thr = self.num_seq[self.seq_idx]                                      # (..., in)
        w_bit = torch.gt(self.w_prob, thr.unsqueeze(-2)).type(self.ntype)     # (..., out, in)
        psum = torch.matmul(w_bit, xf.unsqueeze(-1)).squeeze(-1)              # (..., out)
        if self.seq_idx.shape == x_long.shape:
            # steady state: in-place (long += long, no promotion, not aliased);
            # the first timestep must broadcast the (1,) init up, which add_ cannot
            self.seq_idx.add_(x_long)
        else:
            expanded = self.seq_idx.add(x_long).detach()
            self.seq_idx.resize_as_(expanded).copy_(expanded)

        if self.has_bias:
            # bias bit advances unconditionally, one position per timestep (bool add
            # promotes to ntype, same values as an explicit cast)
            b_bit = torch.gt(self.b_prob, self.num_seq[(self.timestep_cur - 1) % self.len])
            psum = psum + b_bit

        if self._is_bipolar:
            # input-0 path: complemented weight bit against complemented input spike
            thr_inv = self.num_seq[self.seq_idx_inv]                          # (..., in)
            w_bit_inv = torch.le(self.w_prob, thr_inv.unsqueeze(-2)).type(self.ntype)
            psum = psum + torch.matmul(w_bit_inv, (1 - xf).unsqueeze(-1)).squeeze(-1)
            if self.seq_idx_inv.shape == x_long.shape:
                self.seq_idx_inv.add_(1 - x_long)
            else:
                expanded = self.seq_idx_inv.add(1 - x_long).detach()
                self.seq_idx_inv.resize_as_(expanded).copy_(expanded)

        return self.acc(psum, entry=self.entry, dim=None)                     # (..., out)

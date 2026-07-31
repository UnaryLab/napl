import torch
import math

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger


def _init_linear_params(module, in_features, out_features, bias, weight_ext, bias_ext):
    """Give `module` nn.Linear-style learnable weight/bias (or adopt external tensors)."""
    module.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
    torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    if bias:
        module.bias = torch.nn.Parameter(torch.empty(out_features))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(module.bias, -bound, bound)
    else:
        module.bias = None
    if weight_ext is not None:
        assert tuple(weight_ext.shape) == (out_features, in_features), \
            logger.error(f'weight_ext shape {tuple(weight_ext.shape)} != ({out_features}, {in_features}).')
        module.weight.data = weight_ext.clone().type(module.weight.dtype)
    if bias and bias_ext is not None:
        assert tuple(bias_ext.shape) == (out_features,), \
            logger.error(f'bias_ext shape {tuple(bias_ext.shape)} != ({out_features},).')
        module.bias.data = bias_ext.clone().type(module.bias.dtype)


def _linear_ste_grads(ctx, grad_output):
    """
    Straight-through gradient shared by the binary-domain linear kernels: the forward is
    an approximate/quantized matmul, but the backward is the exact linear gradient so the
    layer trains like a normal nn.Linear.
    """
    input, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None
    if ctx.needs_input_grad[0]:
        grad_input = grad_output.matmul(weight)
    if ctx.needs_input_grad[1]:
        grad_weight = grad_output.t().matmul(input)
    if bias is not None and ctx.needs_input_grad[2]:
        grad_bias = grad_output.sum(0)
    return grad_input, grad_weight, grad_bias


class linear(napl_base):
    """Apply a rate-coded unary fully connected layer one timestep at a time.

    Use this layer when inputs are already spike tensors and weights should be
    encoded on a separate number-sequence dimension. It computes ``W x + b`` bit by bit.
    Each timestep the weights (and bias) are encoded into spikes on a distinct RNG
    dimension from the input (so the operand streams are decorrelated), multiplied with
    the incoming input spikes (XNOR for bipolar, AND for unipolar), and the partial
    products are summed by a scaled unary adder. The decoded output value is the inner
    product divided by `scale` (default in_features + has_bias) so it stays in unary
    range, so it represents ``(W x + b) / scale``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear

        layer = linear(torch.zeros(3, 2), config={"polarity": "bipolar",
                       "timestep": 4, "generator": "sobol"})
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
                'dim': 2,
                'scale': None,
                'width': 12,
            }
        ):
        """Construct the streaming layer from external numeric parameters.

        Args:
            weight: Numeric tensor shaped ``(out_features, in_features)``.
            bias: Optional numeric tensor shaped ``(out_features,)``. Defaults
                to ``None``.
            config: Configuration mapping with these keys:

                * **polarity** - ``"unipolar"`` or ``"bipolar"``. Defaults to
                  ``"bipolar"``.
                * **timestep** - Weight-encoder stream length. Defaults to ``256``.
                * **generator** - Number-sequence generator. Defaults to
                  ``"sobol"``.
                * **dim** - One-based weight Sobol dimension. Defaults to ``2``;
                  bias uses the next dimension.
                * **scale** - Output scaling divisor. ``None`` uses
                  ``in_features + has_bias``. Defaults to ``None``.
                * **width** - Signed accumulator width. Defaults to ``12`` and
                  must satisfy ``2 ** (width - 1) >= in_features + has_bias``.
                * **name** - Optional instance label. Defaults to ``None``.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # lazy import: operation.mul_csg imports module.encoder, so importing operation at
        # module top would create an import cycle with module/__init__.
        from napl.sim.operation import add_any
        from napl.sim.module.encoder import encoder

        assert weight.dim() == 2, logger.error(f'linear weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
        self.weight = weight
        self.out_features, self.in_features = weight.shape
        self.has_bias = bias is not None
        self.entry = self.in_features + (1 if self.has_bias else 0)
        scale = config.get('scale', None)
        self.scale = self.entry if scale is None else scale

        # the scaled accumulator must hold a per-step partial sum up to `entry`; if the
        # accumulator range 2**(width-1) is smaller it saturates and silently returns
        # near-maximal error, so reject that configuration outright.
        width = config.get('width', 12)
        assert 2 ** (width - 1) >= self.entry, logger.error(
            f'linear accumulator width <{width}> too small for fan-in <{self.entry}>: '
            f'2**(width-1) must be >= entry or partial sums saturate. Increase width.')

        dim = config.get('dim', 2)
        # weight encoder on its own RNG dim; input is encoded by the caller on a different dim.
        # NB: decorrelation-by-dim only works for the sobol family; lfsr/tc/temporal ignore
        # dim and yield identical sequences across operands, biasing the result.
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'linear decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim (identical sequences across '
                f'operands). Use a sobol-family generator, or decorrelate the input and weight '
                f'streams by distinct seeds.')
        self.w_encoder = encoder({'polarity': self.polarity, 'timestep': config['timestep'],
                                  'generator': config['generator'], 'dim': dim})
        self.acc = add_any({'polarity': self.polarity, 'scale': self.scale, 'width': config.get('width', 12)})

        if self.has_bias:
            self.bias = bias
            self.b_encoder = encoder({'polarity': self.polarity, 'timestep': config['timestep'],
                                      'generator': config['generator'], 'dim': dim + 1})

    def _reset(self):
        """Reset state owned directly by the layer.

        This class has no extra local state. The inherited ``reset()`` method
        resets the weight and bias encoders and the unary adder.
        """
        pass

    def forward(self, input_spike):
        """Process one input-spike timestep.

        Args:
            input_spike: ``0``/``1`` tensor whose last dimension is
                ``in_features``. Leading dimensions are preserved.

        Returns:
            Output spike tensor with the last dimension replaced by
            ``out_features``.

        The call advances this layer and its registered streaming children and
        updates the adder accumulator. External weight and bias tensors are not
        modified.
        """
        # input_spike: (..., in_features) spike tensor for the current timestep
        w_spike = self.w_encoder(self.weight)                       # (out_features, in_features)
        xf = input_spike.type(self.ntype)
        wf = w_spike.type(self.ntype)
        # partial sum of the AND (unipolar) / XNOR (bipolar) spike products, without
        # materializing the (..., out, in) elementwise product; bit-exact (small integers)
        psum = torch.matmul(xf, wf.t())                             # (..., out_features)
        if self.polarity == 'bipolar':
            psum = 2 * psum - xf.sum(-1, keepdim=True) - wf.sum(-1) + self.in_features
        if self.has_bias:
            psum = psum + self.b_encoder(self.bias).type(self.ntype)  # bias spike joins the sum
        return self.acc(psum, entry=self.entry, dim=None)           # (..., out_features)


class linear_pc(napl_base):
    """Return the per-timestep parallel count of a unary linear product.

    Use this streaming layer when downstream logic needs the raw product count
    rather than a scaled output bitstream. It returns the per-timestep binary inner-product
    count of the input spikes against freshly encoded weight spikes, before any accumulation
    into a bitstream. This is the `linear` partial sum without its scaled unary adder.

    Each timestep the weights (and bias) are encoded into spikes on a distinct RNG dimension
    from the input (decorrelated operands). For unipolar this returns the AND-count
    sum(input & weight) (+ bias spike); for bipolar it returns the XNOR-count
    sum(input == weight) (+ bias spike, added on the input-1 path only, matching FSULinearPC).
    The count per timestep lies in [0, entry] with entry = in_features + has_bias; accumulating
    the count over T timesteps and dividing by T recovers the unipolar inner product directly,
    or the bipolar inner product as ``2 * mean - entry``. This class matches
    UnarySim ``FSULinearPC``.

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
        # lazy import: operation.mul_csg imports module.encoder, so importing at module top would
        # create an import cycle with module/__init__.
        from napl.sim.module.encoder import encoder

        assert weight.dim() == 2, logger.error(f'linear_pc weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
        self.weight = weight
        self.out_features, self.in_features = weight.shape
        self.has_bias = bias is not None
        self.entry = self.in_features + (1 if self.has_bias else 0)

        dim = config.get('dim', 2)
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'linear_pc decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim (identical sequences across '
                f'operands). Use a sobol-family generator, or decorrelate the input and weight '
                f'streams by distinct seeds.')
        self.w_encoder = encoder({'polarity': self.polarity, 'timestep': config['timestep'],
                                  'generator': config['generator'], 'dim': dim})
        if self.has_bias:
            self.bias = bias
            self.b_encoder = encoder({'polarity': self.polarity, 'timestep': config['timestep'],
                                      'generator': config['generator'], 'dim': dim + 1})

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
        # input_spike: (..., in_features) spike tensor for the current timestep
        w_spike = self.w_encoder(self.weight)                       # (out_features, in_features)
        xf = input_spike.type(self.ntype)
        wf = w_spike.type(self.ntype)
        # AND-count of the input-1 path, without materializing the (..., out, in) product
        and_count = torch.matmul(xf, wf.t())                        # (..., out_features)
        pc = and_count
        if self.has_bias:
            # bias spike joins the input-1 path only (matches FSULinearPC bias placement)
            pc = pc + self.b_encoder(self.bias).type(self.ntype)
        if self.polarity == 'bipolar':
            # XNOR-count: add the input-0 path sum((1-input)&(1-weight)). Computed
            # algebraically from the AND-count instead of a second matmul over the
            # materialized (1-input)/(1-weight) tensors (bit-exact integer identity:
            # sum((1-x)(1-w)) = in - sum(x) - sum(w) + sum(xw)), saving a matmul and two
            # full-tensor allocations per timestep.
            input0 = self.in_features - xf.sum(-1, keepdim=True) - wf.sum(-1) + and_count
            pc = pc + input0
        return pc                                                   # (..., out_features)


class _linear_fxp_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, rshift_i, rshift_w, rshift_o, max_abs_i, max_abs_w):
        ctx.save_for_backward(input, weight, bias)
        bot_i, top_i = 1 - max_abs_i, max_abs_i - 1
        i_round = pow2_rshift(input, rshift_i)
        i_round.round_().clamp_(bot_i, top_i)
        bot_w, top_w = 1 - max_abs_w, max_abs_w - 1
        w_round = pow2_rshift(weight, rshift_w)
        w_round.round_().clamp_(bot_w, top_w)
        output = torch.matmul(i_round, w_round.t())
        output = pow2_rshift(output, rshift_o)
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return _linear_ste_grads(ctx, grad_output) + (None, None, None, None, None)


class linear_fxp(napl_base):
    """Apply a trainable fixed-point approximation of ``torch.nn.Linear``.

    Use this single-shot layer for quantization-aware evaluation or training. It dynamically scales input and weight
    to widthi/widthw-bit fixed point (via rshift_offset over their quantile magnitude),
    matmul, then shift the output back. Single-shot; trains via STE (exact linear
    gradient) and approximates ``nn.Linear`` within the quantization bound.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_fxp

        layer = linear_fxp(2, 3)
        output = layer(torch.zeros(1, 2))
    """
    streaming = False
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            weight_ext=None,
            bias_ext=None,
            config={
                'widthi': 8,
                'quantilei': 1,
                'widthw': 8,
                'quantilew': 1,
                'rounding': 'round',
            }
        ):
        """Construct the fixed-point layer and initialize trainable parameters.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Create a trainable bias when ``True``. Defaults to ``True``.
            weight_ext: Optional initial weight tensor shaped
                ``(out_features, in_features)``. Defaults to ``None``.
            bias_ext: Optional initial bias tensor shaped ``(out_features,)``.
                Used only when **bias** is ``True``. Defaults to ``None``.
            config: Configuration mapping with **widthi** and **widthw** (input
                and weight widths, both default ``8``), **quantilei** and
                **quantilew** (scaling quantiles, both default ``1``), and
                **rounding** (rounding mode, default ``"round"``). **name** is an
                optional instance label and defaults to ``None``.
        """
        super().__init__(config, [])
        self.in_features, self.out_features = in_features, out_features
        self.widthi = config.get('widthi', 8)
        self.widthw = config.get('widthw', 8)
        self.quantilei = config.get('quantilei', 1)
        self.quantilew = config.get('quantilew', 1)
        self.rounding = config.get('rounding', 'round').lower()
        self.max_abs_i = 2 ** self.widthi
        self.max_abs_w = 2 ** self.widthw
        _init_linear_params(self, in_features, out_features, bias, weight_ext, bias_ext)

    def _reset(self):
        """Reset local execution state.

        This single-shot layer has no mutable run state, so the hook returns
        ``None`` without changing trainable parameters.
        """
        pass

    def forward(self, input):
        """Apply the fixed-point linear approximation.

        Args:
            input: Numeric tensor shaped ``(batch, in_features)``.

        Returns:
            Numeric tensor shaped ``(batch, out_features)``.

        The call computes dynamic shifts from the input and weight. It does not
        change persistent state or ``timestep_cur``; the custom backward uses the
        straight-through linear gradient.
        """
        rshift_i, rshift_w, _ = rshift_offset(input, self.weight, self.widthi - 1, self.widthw - 1,
                                              self.rounding, self.quantilei, self.quantilew)
        rshift_o = 0 - rshift_i - rshift_w
        return _linear_fxp_fn.apply(input, self.weight, self.bias,
                                    rshift_i, rshift_w, rshift_o, self.max_abs_i, self.max_abs_w)


def _hub_rng_seq(width, rng='sobol'):
    """
    Integer RNG sequence of length 2**width in [0, 2**width).
    Used to build the HUB unary-multiplication value map.
    """
    seq_len = 2 ** width
    rng = rng.lower()
    if rng in ('sobol', 'rc'):
        seq = torch.quasirandom.SobolEngine(1).draw(seq_len)[:, 0].view(seq_len) * seq_len
    elif rng in ('race', 'tc'):
        seq = torch.tensor([x / seq_len for x in range(seq_len)]) * seq_len            # ascending
    elif rng in ('race10', 'tc10'):
        seq = torch.flip(torch.tensor([x / seq_len for x in range(seq_len)]) * seq_len, [0])
    else:
        seq = torch.quasirandom.SobolEngine(1).draw(seq_len)[:, 0].view(seq_len) * seq_len
    return seq.floor()


def _build_hub_map(widthi, widthw, rngi, rngw, ntype):
    """
    Build the HUB unary-multiplication value map: mapcbsg[i_level, w_level] is the
    bitstream AND-count for an input of magnitude i_level and a weight of magnitude
    w_level, under the chosen RNGs (sign-magnitude, so cycle_max = 2**(width-1)).
    Returns (cycle_max, mapcbsg). Shared by linear_hub and conv_hub. Requires widthi==widthw.
    """
    cmax = 2 ** (max(widthi, widthw) - 1)
    rngctler = _hub_rng_seq(widthi - 1, rngi)   # controller = input
    rngctlee = _hub_rng_seq(widthw - 1, rngw)   # controllee = weight
    levels = torch.arange(cmax, dtype=torch.float).unsqueeze(1)               # (cmax,1)
    ctler_bit = torch.gt(levels.expand(cmax, cmax), rngctler.unsqueeze(0))    # [i,j] = i > rngctler[j]
    mapctler = torch.sum(ctler_bit, 1).type(torch.long)                      # one-count per input level
    ctlee_bit = torch.gt(levels.expand(cmax, cmax), rngctlee.unsqueeze(0))    # [i,j] = i > rngctlee[j]
    mapcbsg = torch.empty(cmax, cmax, dtype=torch.long)
    for c in range(cmax):
        # over the input-on cycles (first mapctler[c] positions), count weight-on overlaps
        mapcbsg[c] = torch.sum(ctlee_bit[:, 0:mapctler[c]], 1)
    return cmax, mapcbsg.type(ntype)


class _linear_hub_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, rshift_i, rshift_w, rshift_o, cycle, mapcbsg):
        ctx.save_for_backward(input, weight, bias)
        assert input.dim() == 2, logger.error('linear_hub input needs 2 dims (batch, in_features).')
        # quantize |input|, |weight| to unary levels [0, cycle); keep their signs separately
        buf_i = pow2_rshift(input, rshift_i).unsqueeze(1).round().abs().clamp(0, cycle - 1).type(torch.long)   # (batch,1,in)
        buf_w = pow2_rshift(weight, rshift_w).unsqueeze(0).round().abs().clamp(0, cycle - 1).type(torch.long)  # (1,out,in)
        act_input = torch.sign(input).unsqueeze(1)                       # (batch,1,in)
        act_wght = torch.sign(weight).unsqueeze(0)                       # (1,out,in)
        # look up the unary product magnitude for each (input level, weight level), apply weight
        # sign; advanced indexing broadcasts (batch,1,in) against (1,out,in) to (batch,out,in)
        # without materializing the batch-expanded index/sign tensors.
        prod = mapcbsg[buf_i, buf_w].type(act_wght.dtype) * act_wght    # (batch,out,in)
        output = torch.matmul(act_input, prod.transpose(1, 2))          # (batch,1,out)
        output = pow2_rshift(output, rshift_o).squeeze(1)               # (batch,out)
        if bias is not None:
            output = output + bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return _linear_ste_grads(ctx, grad_output) + (None, None, None, None, None)


class linear_hub(napl_base):
    """Apply a hybrid unary-binary approximation of ``torch.nn.Linear``.

    Use this single-shot trainable layer when products should use a precomputed
    unary multiplication value map while the interface remains numeric. Input and weight are
    quantized to sign-magnitude fixed point, and each absolute input-weight product is looked
    up from a precomputed value map that emulates the unary (bitstream-AND) multiplication
    under the chosen RNG. Single-shot; trains via STE. Approximates nn.Linear
    within the unary-multiplication bound. Rate coding, sign-magnitude.
    Requires ``widthi == widthw``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_hub

        layer = linear_hub(2, 3, config={"widthi": 4, "widthw": 4,
                                        "cycle": 8})
        output = layer(torch.zeros(1, 2))
    """
    streaming = False
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            weight_ext=None,
            bias_ext=None,
            config={
                'widthi': 8,
                'rngi': 'sobol',
                'quantilei': 1,
                'widthw': 8,
                'rngw': 'sobol',
                'quantilew': 1,
                'cycle': 128,
                'rounding': 'round',
            }
        ):
        """Construct the HUB layer and its unary-product lookup map.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Create a trainable bias when ``True``. Defaults to ``True``.
            weight_ext: Optional initial weight tensor. Defaults to ``None``.
            bias_ext: Optional initial bias tensor. Defaults to ``None``.
            config: Configuration mapping with these keys:

                * **widthi**, **widthw** - Equal sign-magnitude widths. Both
                  default to ``8``.
                * **rngi**, **rngw** - Input and weight RNG names. Both default
                  to ``"sobol"``; ``"rc"``, ``"race"``, ``"tc"``,
                  ``"race10"``, and ``"tc10"`` follow the implemented map rules.
                * **quantilei**, **quantilew** - Dynamic-scaling quantiles. Both
                  default to ``1``.
                * **cycle** - Active unary cycles, capped at ``2 ** (widthi - 1)``.
                  The declared default is ``128``; ``None`` selects the cap.
                * **rounding** - Dynamic-scaling rounding mode. Defaults to
                  ``"round"``.
                * **name** - Optional instance label. Defaults to ``None``.

        The value map is persistent non-trainable state; weights and optional bias
        are trainable parameters.
        """
        super().__init__(config, [])
        self.in_features, self.out_features = in_features, out_features
        self.widthi = config.get('widthi', 8)
        self.widthw = config.get('widthw', 8)
        assert self.widthi == self.widthw, \
            logger.error(f'linear_hub requires widthi == widthw (got {self.widthi}, {self.widthw}).')
        self.rngi = config.get('rngi', 'sobol').lower()
        self.rngw = config.get('rngw', 'sobol').lower()
        self.quantilei = config.get('quantilei', 1)
        self.quantilew = config.get('quantilew', 1)
        self.rounding = config.get('rounding', 'round').lower()
        # signmag is always True: one cycle bit is the sign, so cycle_max = 2**(width-1)
        self.cycle_max, mapcbsg = _build_hub_map(self.widthi, self.widthw, self.rngi, self.rngw, self.ntype)
        cycle_cfg = config.get('cycle', None)
        self.cycle_act = self.cycle_max if cycle_cfg is None else min(cycle_cfg, self.cycle_max)
        self.register_buffer('mapcbsg', mapcbsg)

        _init_linear_params(self, in_features, out_features, bias, weight_ext, bias_ext)

    def _reset(self):
        """Reset local execution state.

        This single-shot layer has no mutable run state. The lookup map and
        trainable parameters are unchanged.
        """
        pass

    def forward(self, input):
        """Apply the HUB linear approximation.

        Args:
            input: Two-dimensional numeric tensor shaped
                ``(batch, in_features)``.

        Returns:
            Numeric tensor shaped ``(batch, out_features)``.

        The call reads the lookup map and computes dynamic shifts without changing
        persistent state or ``timestep_cur``. Backpropagation uses a
        straight-through linear gradient.
        """
        rshift_i, rshift_w, rshift_o = rshift_offset(input, self.weight, self.widthi - 1, self.widthw - 1,
                                                     self.rounding, self.quantilei, self.quantilew)
        return _linear_hub_fn.apply(input, self.weight, self.bias,
                                    rshift_i, rshift_w, rshift_o, self.cycle_act, self.mapcbsg)


def _tlut_decompose(mag, widtht, degree, cycle_neg, cycle_pos):
    """
    Temporal LUT decomposition: split a truncated fixed-point magnitude tensor into a
    sum of `degree` `widtht`-bit temporal digits (each clamped to the run cycle range).
    Returns the recomposed magnitude. Shared by the TLUT forward modes.
    """
    out = torch.zeros_like(mag)
    for _ in range(degree):
        mag = pow2_rshift(mag, widtht)
        frac = torch.frac(mag)
        mag = torch.trunc(mag)
        frac = pow2_lshift(frac, widtht).clamp(cycle_neg + 1, cycle_pos - 1)
        out = pow2_rshift(frac, widtht) + pow2_rshift(out, widtht)
    return out


class _linear_tlut_fxpfxp_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, temporal, widthi, widthw, widtht, degree, delta,
                cycle_pos, cycle_neg, rounding, quantilei, quantilew):
        ctx.save_for_backward(input, weight, bias)
        in_fp = input.detach().clone().to(torch.float)
        w_fp = weight.detach().clone().to(torch.float)
        rshift_i, rshift_w, _ = rshift_offset(in_fp, w_fp, widthi, widthw, rounding, quantilei, quantilew)
        in_fp = torch.trunc(pow2_rshift(in_fp, rshift_i).clamp(-2 ** widthi + 1, 2 ** widthi - 1))
        w_fp = torch.trunc(pow2_rshift(w_fp, rshift_w).clamp(-2 ** widthw + 1, 2 ** widthw - 1))
        if temporal in ('i', 'input'):
            in_new = _tlut_decompose(in_fp, widtht, degree, cycle_neg, cycle_pos)
            input_new = pow2_lshift(in_new, delta + widthi + rshift_i).type(weight.dtype)
            weight_new = pow2_lshift(w_fp, rshift_w).type(weight.dtype)
        else:
            w_new = _tlut_decompose(w_fp, widtht, degree, cycle_neg, cycle_pos)
            input_new = pow2_lshift(in_fp, rshift_i).type(input.dtype)
            weight_new = pow2_lshift(w_new, delta + widthw + rshift_w).type(input.dtype)
        output = torch.matmul(input_new, weight_new.t())
        if bias is not None:
            output = output + bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return _linear_ste_grads(ctx, grad_output) + (None,) * 11


class _linear_tlut_fxpfp_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, temporal, width, widtht, degree, delta,
                cycle_pos, cycle_neg, rounding, quantilei, quantilew):
        ctx.save_for_backward(input, weight, bias)
        in_fp = input.detach().clone().to(torch.float)
        w_fp = weight.detach().clone().to(torch.float)
        rshift_i, rshift_w, _ = rshift_offset(in_fp, w_fp, width, width, rounding, quantilei, quantilew)
        if temporal in ('i', 'input'):
            in_fp = torch.trunc(pow2_rshift(in_fp, rshift_i).clamp(-2 ** width + 1, 2 ** width - 1))
            in_new = _tlut_decompose(in_fp, widtht, degree, cycle_neg, cycle_pos)
            input_new = pow2_lshift(in_new, delta + width + rshift_i).type(weight.dtype)
            weight_new = weight
        else:
            w_fp = torch.trunc(pow2_rshift(w_fp, rshift_w).clamp(-2 ** width + 1, 2 ** width - 1))
            w_new = _tlut_decompose(w_fp, widtht, degree, cycle_neg, cycle_pos)
            input_new = input
            weight_new = pow2_lshift(w_new, delta + width + rshift_w).type(input.dtype)
        output = torch.matmul(input_new, weight_new.t())
        if bias is not None:
            output = output + bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return _linear_ste_grads(ctx, grad_output) + (None,) * 10


class _linear_tlut_fpfp_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, temporal, width, widtht, degree, delta, cycle_pos, cycle_neg):
        ctx.save_for_backward(input, weight, bias)
        dtype = input.dtype
        src = (input if temporal in ('i', 'input') else weight).detach().clone().to(torch.float)
        try:
            mantissa, exponent = torch.frexp(src)
        except NotImplementedError:
            # frexp has no MPS kernel; compute it on cpu and move the results back (bit-exact).
            mantissa, exponent = torch.frexp(src.cpu())
            mantissa, exponent = mantissa.to(src.device), exponent.to(src.device)
        mantissa = _tlut_decompose(pow2_lshift(mantissa, width), widtht, degree, cycle_neg, cycle_pos)
        mantissa = pow2_lshift(mantissa, delta)
        recomposed = torch.ldexp(mantissa, exponent).type(dtype)
        if temporal in ('i', 'input'):
            input_new, weight_new = recomposed, weight
        else:
            input_new, weight_new = input, recomposed
        output = torch.matmul(input_new, weight_new.t())
        if bias is not None:
            output = output + bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return _linear_ste_grads(ctx, grad_output) + (None,) * 7


_TLUT_FP_WIDTH = {'bfloat16': 8, 'float16': 11, 'float32': 24}


class linear_tlut(napl_base):
    """Apply a temporal-LUT approximation of ``torch.nn.Linear``.

    Use this single-shot trainable layer to decompose either the input or weight
    into temporal digits while retaining a numeric interface. The chosen operand (input or
    weight) is decomposed into a sum of widtht-bit temporal digits and accumulated, while
    the other operand stays fixed-point (fxp) or floating-point (fp). Three modes follow
    from the (formati, formatw) pair: fxpfxp, fxpfp, fpfp. Single-shot; trains via
    STE and approximates ``nn.Linear`` within the temporal-decomposition bound.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_tlut

        layer = linear_tlut(2, 3, config={"temporal": "i", "widtht": 4,
                                         "formati": "fxp", "widthi": 8,
                                         "formatw": "fxp", "widthw": 8})
        output = layer(torch.zeros(1, 2))
    """
    streaming = False
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            weight_ext=None,
            bias_ext=None,
            config={
                'temporal': 'i',
                'widtht': 4,
                'formati': 'fxp',
                'widthi': 8,
                'quantilei': 1,
                'formatw': 'fxp',
                'widthw': 8,
                'quantilew': 1,
                'cycle': None,
                'rounding': 'round',
            }
        ):
        """Construct the temporal-LUT layer and select its execution mode.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Create a trainable bias when ``True``. Defaults to ``True``.
            weight_ext: Optional initial weight tensor. Defaults to ``None``.
            bias_ext: Optional initial bias tensor. Defaults to ``None``.
            config: Configuration mapping with these keys:

                * **temporal** - ``"i"`` or ``"input"`` to decompose inputs;
                  ``"w"`` or ``"weight"`` to decompose weights. Defaults to
                  ``"i"``.
                * **widtht** - Bits per temporal digit. Defaults to ``4``.
                * **formati**, **formatw** - Operand formats. ``"fxp"`` selects
                  fixed point; floating formats must be keys in the implemented
                  width map: ``"bfloat16"``, ``"float16"``, or ``"float32"``.
                  Both default to ``"fxp"``.
                * **widthi**, **widthw** - Fixed-point widths. Both default to ``8``.
                * **quantilei**, **quantilew** - Scaling quantiles. Both default
                  to ``1``.
                * **cycle** - Active cycles, capped at ``2 ** widtht``. ``None``
                  selects the cap and is the default.
                * **rounding** - Fixed-point rounding mode. Defaults to ``"round"``.
                * **name** - Optional instance label. Defaults to ``None``.
        """
        super().__init__(config, [])
        self.in_features, self.out_features = in_features, out_features
        self.temporal = config.get('temporal', 'i').lower()
        self.widtht = config.get('widtht', 4)
        self.formati = config.get('formati', 'fxp').lower()
        self.formatw = config.get('formatw', 'fxp').lower()
        self.widthi = config.get('widthi', 8)
        self.widthw = config.get('widthw', 8)
        self.quantilei = config.get('quantilei', 1)
        self.quantilew = config.get('quantilew', 1)
        self.rounding = config.get('rounding', 'round').lower()
        assert self.temporal in ('i', 'input', 'w', 'weight'), \
            logger.error(f"linear_tlut 'temporal' must be one of ['i','input','w','weight'], got {self.temporal}.")

        if self.formati == 'fxp' and self.formatw == 'fxp':
            self.mode = 'fxpfxp'
        elif self.formati != 'fxp' and self.formatw != 'fxp':
            self.mode = 'fpfp'
        else:
            self.mode = 'fxpfp'

        self.cycle_max = 2 ** self.widtht
        cycle_cfg = config.get('cycle', None)
        self.cycle_act = self.cycle_max if cycle_cfg is None else min(cycle_cfg, self.cycle_max)
        self.widthi_mag = self.widthi - 1
        self.widthw_mag = self.widthw - 1

        # bit-width of the temporal operand's magnitude
        if self.temporal in ('i', 'input'):
            fmt = self.formati
            self.width = self.widthi - 1 if fmt == 'fxp' else _TLUT_FP_WIDTH[fmt]
        else:
            fmt = self.formatw
            self.width = self.widthw - 1 if fmt == 'fxp' else _TLUT_FP_WIDTH[fmt]
        self.degree = int(math.ceil(self.width / self.widtht))
        self.delta = int(self.degree * self.widtht - self.width)

        _init_linear_params(self, in_features, out_features, bias, weight_ext, bias_ext)

    def _reset(self):
        """Reset local execution state.

        This single-shot layer has no mutable run state, so the hook returns
        ``None`` without changing trainable parameters.
        """
        pass

    def forward(self, input):
        """Apply the selected temporal-LUT linear approximation.

        Args:
            input: Numeric tensor shaped ``(batch, in_features)``.

        Returns:
            Numeric tensor shaped ``(batch, out_features)``.

        The call does not change persistent state or ``timestep_cur``. The
        selected custom autograd path supplies a straight-through linear gradient.
        """
        cp, cn = self.cycle_act, -self.cycle_act
        if self.mode == 'fxpfxp':
            return _linear_tlut_fxpfxp_fn.apply(input, self.weight, self.bias, self.temporal,
                                                self.widthi_mag, self.widthw_mag, self.widtht, self.degree,
                                                self.delta, cp, cn, self.rounding, self.quantilei, self.quantilew)
        elif self.mode == 'fxpfp':
            return _linear_tlut_fxpfp_fn.apply(input, self.weight, self.bias, self.temporal,
                                               self.width, self.widtht, self.degree, self.delta,
                                               cp, cn, self.rounding, self.quantilei, self.quantilew)
        else:
            return _linear_tlut_fpfp_fn.apply(input, self.weight, self.bias, self.temporal,
                                              self.width, self.widtht, self.degree, self.delta, cp, cn)

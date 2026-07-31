import torch
import math

from napl.sim.base import napl_base
from loguru import logger


class linear_gaines2(napl_base):
    """Apply a streaming Gaines ``gMUL + uADD`` fully connected layer.

    Use this variant when each weight column needs an independent Sobol dimension
    and addition should use either a unary accumulator or a direct output tracker. It computes ``W x + b`` bit by
    bit. Gaines multiplication streams each weight column j against its own Sobol
    dimension j+1 (bias on dimension in_features+1), so the partial products across the
    fan-in are mutually decorrelated; per timestep the AND-count (unipolar) / XNOR-count
    (bipolar) of the input spikes against the weight spikes is accumulated by a unary
    adder. With config 'scaled': True (default) the scaled adder emits acc >= entry, so
    the decoded output represents (W x + b) / entry with entry = in_features + has_bias.
    With 'scaled': False the non-scaled Gaines output stage subtracts the accumulation
    offset and emits a spike whenever the accumulator leads the count of spikes already
    emitted, so the decoded output tracks clamp(W x + b, -1, 1) directly.
    This class matches UnarySim ``GainesLinear2``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_gaines2

        layer = linear_gaines2(torch.zeros(3, 2),
                               config={"polarity": "bipolar", "timestep": 4,
                                       "generator": "sobol", "scaled": True})
        output_spike = layer(torch.ones(1, 2))

    References
    ----------
    B. R. Gaines, *Stochastic Computing Systems*.
    """
    def __init__(
            self,
            weight,
            bias=None,
            config={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
                'scaled': True,
                'width': 12,
            }
        ):
        """Construct the Gaines layer and precompute weight spike matrices.

        Args:
            weight: Numeric tensor shaped ``(out_features, in_features)``.
            bias: Optional numeric tensor shaped ``(out_features,)``. Defaults to
                ``None``.
            config: Configuration mapping with **polarity** (default
                ``"bipolar"``), **timestep** (positive stream length, default
                ``256``), **generator** (must be ``"sobol"``, ``"rc"``, or
                ``"rate"``; default ``"sobol"``), **scaled** (default ``True``),
                and **width** (scaled-adder width, default ``12``). **name** is an
                optional instance label and defaults to ``None``.

        In scaled mode, **width** must satisfy
        ``2 ** (width - 1) >= in_features + has_bias``.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # lazy import: operation.mul_csg imports module.encoder, so importing operation at
        # module top would create an import cycle with module/__init__.
        from napl.sim.operation import add_any

        assert weight.dim() == 2, logger.error(f'linear_gaines2 weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
        self.weight = weight
        self.out_features, self.in_features = weight.shape
        self.has_bias = bias is not None
        self.entry = self.in_features + (1 if self.has_bias else 0)
        self.scaled = config.get('scaled', True)

        timestep = config['timestep']
        assert timestep > 0, logger.error(f'Invalid timestep: <{timestep}>; legal values: a positive integer.')
        self.len = 2 ** math.ceil(math.log2(timestep))

        # gMUL needs one independent sequence per weight column; only the sobol family
        # provides decorrelated per-dimension sequences.
        generator = config['generator'].lower()
        assert generator in ['sobol', 'rc', 'rate'], logger.error(
            f'linear_gaines2 requires a sobol-family generator (per-column weight RNG); got <{generator}>.')

        # weight column j streams against sobol dim j+1, bias against dim in_features+1
        # (the UnarySim RNGMulti/RNG dim assignment), decorrelating the fan-in products.
        # The weight/bias spikes depend only on the timestep, so the whole period is
        # precomputed here (O(len*in*out) ntype memory) instead of regenerated per step.
        w_seq = torch.quasirandom.SobolEngine(self.in_features).draw(self.len).type(self.ntype).to(weight.device)
        w_prob = (weight + 1) / 2 if self.polarity == 'bipolar' else weight
        w_spike = torch.gt(w_prob.type(self.ntype).unsqueeze(0), w_seq.unsqueeze(1)).type(self.ntype)  # (len, out, in)
        count_corr = None
        if self.polarity == 'bipolar':
            # XNOR count identity 2*sum(xw) - sum(x) - sum(w) + in: the +/-1 weight matrix
            # absorbs the 2* and -sum(x) terms into the matmul, the rest is a per-timestep
            # constant; exact reorder (small integers in float32)
            w_mat = 2 * w_spike - 1
            count_corr = self.in_features - w_spike.sum(-1)                                  # (len, out)
        else:
            w_mat = w_spike
        self.register_buffer('w_mat_seq', w_mat.transpose(1, 2).contiguous())  # (len, in, out)
        if self.has_bias:
            self.bias = bias
            b_seq = torch.quasirandom.SobolEngine(self.in_features + 1).draw(self.len)[:, self.in_features].type(self.ntype).to(bias.device)
            b_prob = (bias + 1) / 2 if self.polarity == 'bipolar' else bias
            b_spike = torch.gt(b_prob.type(self.ntype).unsqueeze(0), b_seq.unsqueeze(1)).type(self.ntype)  # (len, out)
            count_corr = b_spike if count_corr is None else count_corr + b_spike
        self.register_buffer('count_corr_seq', count_corr)                              # (len, out)

        if self.scaled:
            # the scaled accumulator must hold a per-step partial sum up to `entry`; if the
            # accumulator range 2**(width-1) is smaller it saturates and silently returns
            # near-maximal error, so reject that configuration outright.
            width = config.get('width', 12)
            assert 2 ** (width - 1) >= self.entry, logger.error(
                f'linear_gaines2 accumulator width <{width}> too small for fan-in <{self.entry}>: '
                f'2**(width-1) must be >= entry or partial sums saturate. Increase width.')
            # uADD with scale == entry has zero bipolar offset: acc += count, emit acc >= entry
            self.acc = add_any({'polarity': self.polarity, 'scale': self.entry, 'width': width})
        else:
            # non-scaled output stage: subtract the offset that maps the spike count to the
            # output probability, then emit whenever the accumulator leads the emitted count
            self.offset = ((self.in_features - 1) / 2 + (0.5 if self.has_bias else 0.0)) \
                if self.polarity == 'bipolar' else 0.0
            self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
            self.register_buffer('out_accumulator', torch.zeros(1, dtype=self.ntype))


    def _reset(self):
        """Clear local accumulators used by non-scaled mode.

        When **scaled** is ``False``, both the value accumulator and emitted-spike
        accumulator return to scalar zero. Precomputed sequences are unchanged.
        """
        if not self.scaled:
            self.accumulator.resize_(1).zero_()
            self.out_accumulator.resize_(1).zero_()


    def forward(self, input_spike):
        """Process one input-spike timestep.

        Args:
            input_spike: ``0``/``1`` tensor whose last dimension is
                ``in_features``.

        Returns:
            Output spike tensor with last dimension ``out_features``.

        The call selects the current precomputed weight matrix, updates the unary
        adder or local non-scaled accumulators, and advances ``timestep_cur``.
        """
        # input_spike: (..., in_features) spike tensor for the current timestep
        t = (self.timestep_cur - 1) % self.len
        xf = input_spike.type(self.ntype)
        # per-timestep AND (unipolar) / XNOR (bipolar) spike-product count: one matmul
        # against the precomputed weight matrix plus its per-timestep constant; bit-exact
        # (small integers)
        count = torch.matmul(xf, self.w_mat_seq[t])                              # (..., out_features)
        if self.count_corr_seq is not None:
            count = count + self.count_corr_seq[t]
        if self.scaled:
            return self.acc(count, entry=self.entry, dim=None)                  # (..., out_features)
        delta = count.sub(self.offset)
        if self.accumulator.shape == delta.shape:
            self.accumulator.add_(delta)
        else:
            accumulator = self.accumulator.add(delta).detach()
            self.accumulator.resize_as_(accumulator).copy_(accumulator)
        output = torch.gt(self.accumulator, self.out_accumulator).type(self.ntype)
        if self.out_accumulator.shape == output.shape:
            self.out_accumulator.add_(output)
        else:
            out_accumulator = self.out_accumulator.add(output).detach()
            self.out_accumulator.resize_as_(out_accumulator).copy_(out_accumulator)
        return output.type(self.stype)

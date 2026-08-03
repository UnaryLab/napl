import torch
import math

from napl.sim.base import napl_base
from loguru import logger


class linear_gaines2(napl_base):
    """Apply a streaming Gaines ``gMUL + uADD`` fully connected layer.

    Use this variant when each weight column needs an independent Sobol dimension
    and addition should use either a unary accumulator or a direct output tracker. It computes ``W x + b`` bit by
    bit. Gaines multiplication streams each weight column ``j`` against Sobol
    dimension ``j + 1``, with bias on dimension ``in_features + 1``, so the partial products across the
    fan-in are mutually decorrelated; per timestep the AND-count (unipolar) / XNOR-count
    (bipolar) of the input spikes against the weight spikes is accumulated by a unary
    adder. With ``scaled=True`` the scaled adder emits ``acc >= entry``, so
    the decoded output represents ``(W x + b) / entry`` with
    ``entry = in_features + has_bias``. With ``scaled=False`` the output stage subtracts the accumulation
    offset and emits a spike whenever the accumulator leads the count of spikes already
    emitted, so the decoded output tracks ``clamp(W x + b, -1, 1)`` directly.
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
        ``2 ** (width - 1) > in_features + has_bias``.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # This import stays local to avoid the module-operation import cycle.
        from napl.sim.operation import add_any

        assert weight.dim() == 2, logger.error(f'linear_gaines2 weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
        #: Numeric weight matrix used to precompute per-column spike tables.
        self.weight = weight
        #: Number of output features produced by the layer.
        self.out_features = weight.shape[0]
        #: Number of input features consumed by the layer.
        self.in_features = weight.shape[1]
        #: Whether an encoded bias contributes to the parallel count.
        self.has_bias = bias is not None
        #: Parallel-count fan-in, including the bias when present.
        self.entry = self.in_features + (1 if self.has_bias else 0)
        #: Whether addition uses the streaming unary accumulator.
        self.scaled = config.get('scaled', True)

        timestep = config['timestep']
        assert timestep > 0, logger.error(f'Invalid timestep: <{timestep}>; legal values: a positive integer.')
        #: Power-of-two period of the precomputed weight spike tables.
        self.len = 2 ** math.ceil(math.log2(timestep))

        # Each weight column requires an independent Sobol dimension.
        generator = config['generator'].lower()
        assert generator in ['sobol', 'rc', 'rate'], logger.error(
            f'linear_gaines2 requires a sobol-family generator (per-column weight RNG); got <{generator}>.')

        # Column j uses Sobol dimension j+1; bias uses dimension in_features+1.
        w_seq = torch.quasirandom.SobolEngine(self.in_features).draw(self.len).type(self.ntype).to(weight.device)
        w_prob = (weight + 1) / 2 if self.polarity == 'bipolar' else weight
        w_spike = torch.gt(w_prob.type(self.ntype).unsqueeze(0), w_seq.unsqueeze(1)).type(self.ntype)
        count_corr = None
        if self.polarity == 'bipolar':
            # XNOR count is 2*sum(xw) - sum(x) - sum(w) + in_features.
            w_mat = 2 * w_spike - 1
            count_corr = self.in_features - w_spike.sum(-1)
        else:
            w_mat = w_spike
        #: Per-timestep weight matrices used for spike-domain matrix products.
        self.w_mat_seq: torch.Tensor
        self.register_buffer('w_mat_seq', w_mat.transpose(1, 2).contiguous())
        if self.has_bias:
            #: Numeric bias vector used to precompute the bias spike table.
            self.bias = bias
            b_seq = torch.quasirandom.SobolEngine(self.in_features + 1).draw(self.len)[:, self.in_features].type(self.ntype).to(bias.device)
            b_prob = (bias + 1) / 2 if self.polarity == 'bipolar' else bias
            b_spike = torch.gt(b_prob.type(self.ntype).unsqueeze(0), b_seq.unsqueeze(1)).type(self.ntype)
            count_corr = b_spike if count_corr is None else count_corr + b_spike
        #: Per-timestep correction that completes bipolar XNOR counts and bias.
        self.count_corr_seq: torch.Tensor
        self.register_buffer('count_corr_seq', count_corr)

        if self.scaled:
            # The signed accumulator range must contain every per-step partial sum.
            width = config.get('width', 12)
            assert 2 ** (width - 1) > self.entry, logger.error(
                f'linear_gaines2 accumulator width <{width}> too small for fan-in <{self.entry}>: '
                f'2**(width-1) must be > entry or partial sums saturate. Increase width.')
            # scale=entry gives zero bipolar offset.
            #: Streaming unary adder used by scaled mode.
            self.acc = add_any({'polarity': self.polarity, 'scale': self.entry, 'width': width})
        else:
            #: Bipolar count offset removed before non-scaled accumulation.
            self.offset = ((self.in_features - 1) / 2 + (0.5 if self.has_bias else 0.0)) \
                if self.polarity == 'bipolar' else 0.0
            #: Running non-scaled value total before spike emission.
            self.accumulator: torch.Tensor
            self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
            #: Running count of spikes emitted by non-scaled mode.
            self.out_accumulator: torch.Tensor
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
        t = (self.timestep_cur - 1) % self.len
        xf = input_spike.type(self.ntype)
        count = torch.matmul(xf, self.w_mat_seq[t])
        if self.count_corr_seq is not None:
            count = count + self.count_corr_seq[t]
        if self.scaled:
            return self.acc(count, entry=self.entry, dim=None)
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

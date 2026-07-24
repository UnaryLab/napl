import torch
import math

from napl.sim.base import napl_base
from loguru import logger


class linear_gaines2(napl_base):
    """
    Streaming Gaines fully-connected layer (gMUL + uADD): y = W x (+ b), computed bit by
    bit. Gaines multiplication streams each weight column j against its own Sobol
    dimension j+1 (bias on dimension in_features+1), so the partial products across the
    fan-in are mutually decorrelated; per timestep the AND-count (unipolar) / XNOR-count
    (bipolar) of the input spikes against the weight spikes is accumulated by a unary
    adder. With config 'scaled': True (default) the scaled adder emits acc >= entry, so
    the decoded output represents (W x + b) / entry with entry = in_features + has_bias.
    With 'scaled': False the non-scaled Gaines output stage subtracts the accumulation
    offset and emits a spike whenever the accumulator leads the count of spikes already
    emitted, so the decoded output tracks clamp(W x + b, -1, 1) directly.
    References: B. R. Gaines, "Stochastic Computing Systems". UnarySim: GainesLinear2.
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
        self.w_mat_seq = torch.nn.Parameter(w_mat.transpose(1, 2).contiguous(), requires_grad=False)  # (len, in, out)
        if self.has_bias:
            self.bias = bias
            b_seq = torch.quasirandom.SobolEngine(self.in_features + 1).draw(self.len)[:, self.in_features].type(self.ntype).to(bias.device)
            b_prob = (bias + 1) / 2 if self.polarity == 'bipolar' else bias
            b_spike = torch.gt(b_prob.type(self.ntype).unsqueeze(0), b_seq.unsqueeze(1)).type(self.ntype)  # (len, out)
            count_corr = b_spike if count_corr is None else count_corr + b_spike
        self.count_corr_seq = None if count_corr is None else \
            torch.nn.Parameter(count_corr, requires_grad=False)                              # (len, out)

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
            self.accumulator = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)
            self.out_accumulator = torch.nn.Parameter(torch.zeros(1, dtype=self.ntype), requires_grad=False)


    def _reset(self):
        if not self.scaled:
            self.accumulator.data = torch.zeros(1, dtype=self.ntype, device=self.accumulator.device)
            self.out_accumulator.data = torch.zeros(1, dtype=self.ntype, device=self.out_accumulator.device)


    def forward(self, input_spike):
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
        self.accumulator.data = self.accumulator.add(count - self.offset)
        output = torch.gt(self.accumulator, self.out_accumulator).type(self.ntype)
        self.out_accumulator.data = self.out_accumulator.add(output)
        return output.type(self.stype)

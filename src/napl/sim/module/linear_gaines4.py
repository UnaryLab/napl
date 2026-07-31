import torch
import math

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger


class linear_gaines4(napl_base):
    """Apply a streaming Gaines layer with per-column number sequences.

    Use this ``gMUL + gADD`` variant to reproduce the LFSR-oriented UnarySim
    ``GainesLinear4`` design. It is a unary fully-connected layer computed bit by bit:
    y = W x (+ b), computed bit by bit. Each timestep the weights (and bias) are encoded
    into spikes on their own decorrelated number sequences (distinct sobol dim / lfsr
    seed from the input), multiplied with the incoming input spikes (AND for unipolar,
    XNOR for bipolar), and the per-timestep parallel count is reduced to one output
    spike by a Gaines adder:

    - scaled (default): the count is compared against a random threshold drawn from a
      2**round(log2(entry))-long sequence (entry = in_features + has_bias), i.e. the
      classic Gaines scaled addition; the decoded output represents roughly
      (W x + b) / entry (exactly E[count]/(2**k - 1) with an lfsr threshold, since the
      lfsr never emits 0 and the compare is >=, the same quirk as the upstream).
    - non-scaled: unipolar ORs the partial products (output spike = count > 0); bipolar
      integrates 2*count - entry into a saturating counter of `depth` bits (init at
      half range) and emits counter > half, a sign-tracking non-scaled sum.

    Weights are rate-coded from the full-precision tensor (the upstream quantizes them
    to `bitwidth` bits first; agreement is within the SC bound). The weight/bias spike
    streams are fixed and L-periodic, so the whole period is precomputed at init and
    forward() only indexes it.
    Weight and bias spike tables are precomputed for the complete sequence period.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_gaines4

        layer = linear_gaines4(torch.zeros(3, 2),
                               config={"polarity": "bipolar", "timestep": 4,
                                       "generator": "lfsr", "scaled": True})
        output_spike = layer(torch.ones(1, 2))

    """
    def __init__(
            self,
            weight,
            bias=None,
            config={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'lfsr',
                'dim': 2,
                'seed': 1,
                'scaled': True,
                'depth': 8,
            }
        ):
        """Construct the layer and precompute its spike tables.

        Args:
            weight: Numeric tensor shaped ``(out_features, in_features)``.
            bias: Optional numeric tensor shaped ``(out_features,)``. Defaults to
                ``None``.
            config: Configuration mapping with **polarity** (default
                ``"bipolar"``), **timestep** (default ``256``), **generator**
                (default ``"lfsr"``), **dim** (first sequence dimension, default
                ``2``), **seed** (first LFSR seed, default ``1``), **scaled**
                (default ``True``), and **depth** (non-scaled bipolar counter bits,
                default ``8``). **name** is an optional instance label and
                defaults to ``None``.

        Scaled mode requires ``in_features + has_bias >= 2``.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # lazy import: operation.mul_csg imports module.encoder, so importing at module top
        # would create an import cycle with module/__init__.
        from napl.sim.module.encoder import gen_num_seq

        assert weight.dim() == 2, logger.error(
            f'linear_gaines4 weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
        self.weight = weight
        self.out_features, self.in_features = weight.shape
        self.has_bias = bias is not None
        self.entry = self.in_features + (1 if self.has_bias else 0)
        self.scaled = config.get('scaled', True)
        self.depth = config.get('depth', 8)

        dim = config.get('dim', 2)
        seed = config.get('seed', 1)
        # decorrelation: sobol-family decorrelates by dim, lfsr by seed (phase shift);
        # other generators yield identical sequences across operands and bias the result.
        if config['generator'].lower() not in ['sobol', 'rc', 'rate', 'lfsr']:
            logger.warning(
                f'linear_gaines4 decorrelates operands via distinct sobol dims / lfsr seeds, '
                f'but generator <{config["generator"]}> does not decorrelate (identical '
                f'sequences across operands). Use a sobol-family or lfsr generator.')

        # per-column weight number sequences (UnarySim RNGMulti/BSGenMulti): every input
        # column gets its own decorrelated sequence. A single shared sequence would make
        # all weight bits comonotone and blow up the parallel-count variance.
        width = math.ceil(math.log2(config['timestep']))
        self.w_len = 2 ** width
        cols = [gen_num_seq({'width': width, 'generator': config['generator'],
                             'dim': dim + j, 'seed': seed + j})
                for j in range(self.in_features)]
        w_num_seq = torch.stack(cols, dim=1).to(weight.device)                       # (L, in)
        w_prob = ((weight + 1) / 2 if self.polarity == 'bipolar' else weight).type(self.ntype)
        # the weight spikes are a fixed L-periodic table, so encode the whole period once,
        # pre-transposed for the matmul, instead of compare+cast every timestep.
        # ponytail: O(L*in*out) memory; re-encode per timestep if layers get big.
        w_spike_t = torch.gt(w_prob.t().unsqueeze(0), w_num_seq.unsqueeze(-1)).type(self.ntype)
        self.register_buffer('w_spike_t', w_spike_t)                                 # (L, in, out)
        if self.polarity == 'bipolar':
            # per-timestep constant of the XNOR identity below: in - sum(w)
            self.register_buffer('w_offset', self.in_features - w_spike_t.sum(1))    # (L, out)
        if self.has_bias:
            self.bias = bias
            # bias spike stream is likewise fixed and L-periodic (the encoder derives the
            # same width from `timestep`): precompute it instead of encoding per timestep.
            b_num_seq = gen_num_seq({'width': width, 'generator': config['generator'],
                                     'dim': dim + self.in_features,
                                     'seed': seed + self.in_features}).to(bias.device)
            b_prob = (bias + 1) / 2 if self.polarity == 'bipolar' else bias
            self.register_buffer('b_spike',
                torch.gt(b_prob.unsqueeze(0), b_num_seq.unsqueeze(1)).type(self.ntype),
            )                                                                        # (L, out)

        if self.scaled:
            assert self.entry >= 2, logger.error(
                f'linear_gaines4 scaled mode needs entry >= 2, got {self.entry}.')
            # Gaines scaled add: compare the parallel count against a random threshold in
            # [0, 2**k) with k = round(log2(entry)), matching GainesLinear4's rng_scale.
            k = round(math.log2(self.entry))
            self.scale_len = 2 ** k
            seq = gen_num_seq({'width': k, 'generator': config['generator'],
                               'dim': dim + self.in_features + 1,
                               'seed': seed + self.in_features + 1})
            # python-float thresholds: skips a per-timestep device tensor select
            self.scale_thresh = (seq.type(self.ntype) * self.scale_len).tolist()
        else:
            # non-scaled bipolar: saturating counter of `depth` bits, init at half range;
            # scalar (1,) accumulator that broadcasts up to the input shape on first use.
            self.cnt_max = 2 ** self.depth - 1
            self.cnt_half = 2 ** (self.depth - 1)
            self.register_buffer('cnt', torch.full((1,), float(self.cnt_half), dtype=self.ntype))

    def _reset(self):
        """Reset the local non-scaled counter.

        When **scaled** is ``False``, the counter returns to half of its configured
        range. Precomputed weight, bias, and threshold sequences are unchanged.
        """
        if not self.scaled:
            self.cnt.resize_(1).fill_(self.cnt_half)

    def forward(self, input_spike):
        """Process one input-spike timestep.

        Args:
            input_spike: ``0``/``1`` tensor whose last dimension is
                ``in_features``.

        Returns:
            Output spike tensor with last dimension ``out_features``.

        The call advances ``timestep_cur`` and may update the non-scaled bipolar
        counter. Precomputed spike tables are read-only.
        """
        # input_spike: (..., in_features) spike tensor for the current timestep
        idx = (self.timestep_cur - 1) % self.w_len
        xf = input_spike.type(self.ntype)
        # AND-count of the input-1 path against this timestep's precomputed weight spikes,
        # without materializing the (..., out, in) product
        pc = torch.matmul(xf, self.w_spike_t[idx])                  # (..., out_features)
        if self.polarity == 'bipolar':
            # XNOR-count: add the input-0 path sum((1-x)&(1-w)) algebraically
            # (bit-exact integer identity: sum((1-x)(1-w)) = in - sum(x) - sum(w) + sum(xw))
            pc = 2 * pc - xf.sum(-1, keepdim=True) + self.w_offset[idx]
        if self.has_bias:
            # bias spike joins the count directly (GainesLinear4 puts the bias in the
            # forward kernel only, never in the inverse kernel)
            pc = pc + self.b_spike[idx]
        if self.scaled:
            thresh = self.scale_thresh[(self.timestep_cur - 1) % self.scale_len]
            return torch.ge(pc, thresh).type(self.stype)
        if self.polarity == 'unipolar':
            return torch.gt(pc, 0).type(self.stype)
        # non-scaled bipolar: integrate 2*pc - entry, saturate, threshold at half range
        delta = 2 * pc - self.entry
        if self.cnt.shape == delta.shape:
            self.cnt.add_(delta).clamp_(0, self.cnt_max)
        else:
            cnt = self.cnt.add(delta).clamp(0, self.cnt_max).detach()
            self.cnt.resize_as_(cnt).copy_(cnt)
        return torch.gt(self.cnt, self.cnt_half).type(self.stype)

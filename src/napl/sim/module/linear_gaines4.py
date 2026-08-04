import torch
import math

from napl.sim.base import napl_base
from loguru import logger


class linear_gaines4(napl_base):
    """Apply a streaming Gaines layer with per-column number sequences.

    Use this ``gMUL + gADD`` variant to reproduce the LFSR-oriented UnarySim
    ``GainesLinear4`` design. It computes ``y = W x (+ b)`` bit by bit. Each
    timestep the weights and bias are encoded into spikes on their own
    decorrelated number sequences, using distinct Sobol dimensions or LFSR
    seeds. They are multiplied with the incoming input spikes using AND for
    unipolar streams and XNOR for bipolar streams, then a Gaines adder reduces
    the per-timestep parallel count to one output spike:

    - ``scaled=True`` compares the count with a threshold from a
      ``2 ** round(log2(entry))``-entry sequence, where
      ``entry = in_features + has_bias``. The decoded output represents roughly
      ``(W x + b) / entry``.
    - ``scaled=False`` emits ``count > 0`` in unipolar mode. Bipolar mode
      integrates ``2 * count - entry`` into a ``depth``-bit saturating counter
      and emits ``counter > half``.

    Weights are rate-coded from the full-precision tensor (the upstream quantizes them
    to ``bitwidth`` bits first; agreement is within the SC bound). The weight and
    bias spike streams are fixed and periodic, so construction precomputes the
    complete period and ``forward()`` only indexes it.

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
        # This import stays local to avoid the module-operation import cycle.
        from napl.sim.module.encoder import gen_num_seq

        assert weight.dim() == 2, logger.error(
            f'linear_gaines4 weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
        #: Numeric weight matrix used to precompute the weight spike table.
        self.weight = weight
        #: Number of output features produced by the layer.
        self.out_features = weight.shape[0]
        #: Number of input features consumed by the layer.
        self.in_features = weight.shape[1]
        #: Whether an encoded bias contributes to the parallel count.
        self.has_bias = bias is not None
        #: Parallel-count fan-in, including the bias when present.
        self.entry = self.in_features + (1 if self.has_bias else 0)
        #: Whether the Gaines adder uses random-threshold scaled addition.
        self.scaled = config.get('scaled', True)
        #: Bit width of the non-scaled bipolar saturating counter.
        self.depth = config.get('depth', 8)

        dim = config.get('dim', 2)
        seed = config.get('seed', 1)
        # Sobol streams decorrelate by dimension; LFSR streams decorrelate by seed.
        if config['generator'].lower() not in ['sobol', 'rc', 'rate', 'lfsr']:
            logger.warning(
                f'linear_gaines4 decorrelates operands via distinct sobol dims / lfsr seeds, '
                f'but generator <{config["generator"]}> does not decorrelate (identical '
                f'sequences across operands). Use a sobol-family or lfsr generator.')

        # Each input column needs a distinct sequence to avoid comonotone weight bits.
        width = math.ceil(math.log2(config['timestep']))
        #: Number of timesteps in the precomputed weight spike period.
        self.w_len = 2 ** width
        cols = [gen_num_seq({'width': width, 'generator': config['generator'],
                             'dim': dim + j, 'seed': seed + j})
                for j in range(self.in_features)]
        w_num_seq = torch.stack(cols, dim=1).to(weight.device)
        w_prob = ((weight + 1) / 2 if self.polarity == 'bipolar' else weight).type(self.ntype)
        w_spike_t = torch.gt(w_prob.t().unsqueeze(0), w_num_seq.unsqueeze(-1)).type(self.ntype)
        #: Precomputed weight spikes indexed by timestep and input feature.
        self.w_spike_t: torch.Tensor
        self.register_buffer('w_spike_t', w_spike_t)
        if self.polarity == 'bipolar':
            # The XNOR correction includes in_features - sum(weight spikes).
            #: Per-timestep correction term for bipolar XNOR product counts.
            self.w_offset: torch.Tensor
            self.register_buffer('w_offset', self.in_features - w_spike_t.sum(1))
        if self.has_bias:
            #: Numeric bias vector used to precompute the bias spike table.
            self.bias = bias
            b_num_seq = gen_num_seq({'width': width, 'generator': config['generator'],
                                     'dim': dim + self.in_features,
                                     'seed': seed + self.in_features}).to(bias.device)
            b_prob = (bias + 1) / 2 if self.polarity == 'bipolar' else bias
            #: Precomputed bias spike vector for each timestep.
            self.b_spike: torch.Tensor
            self.register_buffer('b_spike',
                torch.gt(b_prob.unsqueeze(0), b_num_seq.unsqueeze(1)).type(self.ntype),
            )

        if self.scaled:
            assert self.entry >= 2, logger.error(
                f'linear_gaines4 scaled mode needs entry >= 2, got {self.entry}.')
            # The Gaines threshold spans [0, 2**round(log2(entry))).
            k = round(math.log2(self.entry))
            #: Number of entries in the scaled-adder threshold sequence.
            self.scale_len = 2 ** k
            seq = gen_num_seq({'width': k, 'generator': config['generator'],
                               'dim': dim + self.in_features + 1,
                               'seed': seed + self.in_features + 1})
            #: Precomputed random comparison levels for scaled addition.
            self.scale_thresh = (seq.type(self.ntype) * self.scale_len).tolist()
        else:
            #: Maximum value of the non-scaled bipolar saturating counter.
            self.cnt_max = 2 ** self.depth - 1
            #: Half-range decision threshold and reset value for the counter.
            self.cnt_half = 2 ** (self.depth - 1)
            #: Non-scaled bipolar counter, expanded to the output shape on use.
            self.cnt: torch.Tensor
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
        idx = (self.timestep_cur - 1) % self.w_len
        xf = input_spike.type(self.ntype)
        pc = torch.matmul(xf, self.w_spike_t[idx])
        if self.polarity == 'bipolar':
            # sum((1-x)(1-w)) = in_features - sum(x) - sum(w) + sum(xw).
            pc = 2 * pc - xf.sum(-1, keepdim=True) + self.w_offset[idx]
        if self.has_bias:
            # Bias contributes to the direct path only.
            pc = pc + self.b_spike[idx]
        if self.scaled:
            thresh = self.scale_thresh[(self.timestep_cur - 1) % self.scale_len]
            return torch.ge(pc, thresh).type(self.stype)
        if self.polarity == 'unipolar':
            return torch.gt(pc, 0).type(self.stype)
        # Non-scaled bipolar mode integrates 2*pc-entry around half range.
        delta = 2 * pc - self.entry
        if self.cnt.shape == delta.shape:
            self.cnt.add_(delta).clamp_(0, self.cnt_max)
        else:
            cnt = self.cnt.add(delta).clamp(0, self.cnt_max).detach()
            self.cnt.resize_as_(cnt).copy_(cnt)
        return torch.gt(self.cnt, self.cnt_half).type(self.stype)

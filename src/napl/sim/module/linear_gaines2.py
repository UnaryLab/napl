import torch
import math

from napl.sim.base import napl_base
from loguru import logger


class linear_gaines2(napl_base):
    r"""Apply a streaming Gaines layer with per-column number sequences.

    Use this ``gMUL + gADD`` variant to reproduce the UnarySim ``GainesLinear4``
    design, which is written around LFSR sequences. It computes
    ``y = W x (+ b)`` bit by bit. Each
    timestep the weights and bias are encoded into spikes on their own
    decorrelated number sequences, using distinct Sobol dimensions or LFSR
    seeds. They are multiplied with the incoming input spikes using AND for
    unipolar streams and XNOR for bipolar streams, then a Gaines adder reduces
    the per-timestep parallel count to one output spike:

    - ``scaled=True`` compares the count with a threshold from a
      ``2 ** round(log2(entry))``-entry sequence, where
      ``entry = in_features + has_bias``. Under the default Sobol generator the
      decoded output represents ``(W x + b) / entry``; under LFSR it sits about
      ``1 / entry`` below that.

      The comparison is ``count / L > q[t]``, where ``q`` is the encoder number
      sequence and ``L = scale_len``, so a constant ``count`` produces the output
      rate ``|{t : scale_seq[t] < count}| / L`` for ``scale_seq = round(q L)``.
      Sobol draws each level in ``0 .. L-1`` exactly once, making that rate
      exactly ``count / L``. An LFSR of width ``k`` has period ``2**k - 1`` while
      the encoder draws ``2**k`` entries, so its levels span ``1 .. L-1`` with one
      repeated and ``0`` absent; counts ``0`` and ``1`` then share the rate ``0``
      and the rest are shifted down by roughly one count, which is where the
      ``1 / entry`` offset comes from. The offset is negligible at a realistic
      fan-in but reaches a third of full scale at ``entry = 4``.

      This strict comparison replaces a former ``count >= scale_seq[t]``. Because
      ``count`` and the levels are integers, the two are related exactly by
      ``gt(count, level) == ge(count - 1, level)``, so the strict form recovers
      what the former recovered for a count one lower. Per position they agree
      except where ``count == scale_seq[t]``, so a constant ``count`` loses one
      output spike per period for every position holding that level: exactly one
      under Sobol, and under LFSR none for ``count`` ``0``, two at the repeated
      level, and one elsewhere below ``L``.
    - ``scaled=False`` emits ``count > 0`` in unipolar mode. Bipolar mode
      integrates ``2 * count - entry`` into a ``depth``-bit saturating counter
      and emits ``counter > half``.

    Weights are rate-coded from the full-precision tensor (the upstream quantizes them
    to ``bitwidth`` bits first; agreement is within the SC bound). Construction
    stores only the threshold sequences, one per input feature plus one each for
    the bias and the scaled threshold; ``forward()`` encodes the weights and bias
    for the current timestep, so no stored table grows with the number of output
    features.

    The precise target is the affine map :math:`y = Wx + b`, reachable only up to
    the mode-specific scaling above. With :math:`w_t` the weight spikes encoded
    for timestep :math:`t`, :math:`b_t` the bias spike, :math:`n` the fan-in, and
    :math:`e = n + [\,\text{bias}\,]`, each timestep forms

    .. math::

       c_t = \begin{cases}
       x_t w_t^{\top} + b_t, & \text{unipolar},\\
       2 x_t w_t^{\top} - \sum_j x_{j,t} + \left(n - \sum_j w_{j,t}\right) + b_t,
       & \text{bipolar},
       \end{cases}

    where the parenthesized term is the per-timestep weight offset.
    The Gaines adder then emits

    .. math::

       y_t = \begin{cases}
       \mathbf{1}\{c_t / 2^{k} > q_{t \bmod 2^{k}}\}, & \text{scaled},\\
       \mathbf{1}\{c_t > 0\}, & \text{non-scaled unipolar},\\
       \mathbf{1}\{a_t > 2^{d-1}\}, & \text{non-scaled bipolar},
       \end{cases}
       \qquad
       a_t = \mathrm{clamp}\left(a_{t-1} + 2c_t - e,\; 0,\; 2^{d}-1\right),

    with :math:`a_0 = 2^{d-1}`, :math:`k = \mathrm{round}(\log_2 e)`, and
    :math:`q` the encoder number sequence of the scaled threshold.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_gaines2

        layer = linear_gaines2(torch.zeros(3, 2),
                               config={"polarity": "bipolar", "timestep": 4,
                                       "generator": "sobol", "scaled": True})
        output_spike = layer(torch.ones(1, 2))

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
                (default ``"sobol"``), **dim** (first sequence dimension, default
                ``2``), **seed** (first LFSR seed, default ``1``), **scaled**
                (default ``True``), and **depth** (non-scaled bipolar counter bits,
                default ``8``). **name** is an optional instance label and
                defaults to ``None``.

        Scaled mode requires ``in_features + has_bias >= 2``.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # These imports stay local to avoid the module-operation import cycle.
        from napl.sim.operation import encode, gen_num_seq

        assert weight.dim() == 2, logger.error(
            f'linear_gaines2 weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
        #: Trainable numeric weight matrix encoded to spikes each timestep.
        self.weight = torch.nn.Parameter(weight)
        self._weight_prob_cache = None
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
                f'linear_gaines2 decorrelates operands via distinct sobol dims / lfsr seeds, '
                f'but generator <{config["generator"]}> does not decorrelate (identical '
                f'sequences across operands). Use a sobol-family or lfsr generator.')

        # Each input column needs a distinct sequence to avoid comonotone weight bits.
        width = math.ceil(math.log2(config['timestep']))
        #: Number of timesteps in the periodic weight spike stream.
        self.w_len = 2 ** width
        cols = [gen_num_seq({'width': width, 'generator': config['generator'],
                             'dim': dim + j, 'seed': seed + j})
                for j in range(self.in_features)]
        #: Per-input-feature threshold sequences indexed by timestep.
        self.w_num_seq: torch.Tensor
        self.register_buffer('w_num_seq', torch.stack(cols, dim=1).to(weight.device))
        # Distinct dims or seeds do not guarantee distinct sequences: an LFSR seed
        # reducing to zero aliases onto 2**width - 1, so check what was derived.
        distinct = torch.unique(self.w_num_seq, dim=1).shape[1]
        if distinct < self.in_features:
            logger.warning(
                f'linear_gaines2 derived only {distinct} distinct weight sequences for '
                f'{self.in_features} input features, so some features share one sequence and '
                f'their weight bits are comonotone. With generator <{config["generator"]}> at '
                f'timestep {config["timestep"]} the sequence period is {self.w_len}; reduce '
                f'in_features below that period or raise timestep.')
        if self.has_bias:
            #: Trainable numeric bias vector encoded to spikes each timestep.
            self.bias = torch.nn.Parameter(bias)
            #: Encoder that converts the numeric bias to one spike per timestep.
            self.b_encoder = encode({'polarity': self.polarity, 'timestep': config['timestep'],
                                     'generator': config['generator'],
                                     'dim': dim + self.in_features,
                                     'seed': seed + self.in_features})

        if self.scaled:
            assert self.entry >= 2, logger.error(
                f'linear_gaines2 scaled mode needs entry >= 2, got {self.entry}.')
            # The Gaines threshold spans [0, 2**round(log2(entry))).
            k = round(math.log2(self.entry))
            #: Number of entries in the scaled-adder threshold sequence.
            self.scale_len = 2 ** k
            #: Encoder supplying the scaled-adder threshold comparison.
            self.reference_encode = encode({'polarity': 'unipolar',
                                            'timestep': self.scale_len,
                                            'generator': config['generator'],
                                            'dim': dim + self.in_features + 1,
                                            'seed': seed + self.in_features + 1})
            scaled_levels = self.reference_encode.num_seq.mul(self.scale_len)
            assert torch.allclose(scaled_levels, scaled_levels.round(), atol=1e-9), \
                f'Sequence value off the 1/{self.scale_len} grid; the count-scale view would not be exact.'
            #: Count-scale view of the encoder sequence, kept for inspection.
            self.scale_seq: torch.Tensor
            self.register_buffer('scale_seq', scaled_levels.round())
        else:
            #: Maximum value of the non-scaled bipolar saturating counter.
            self.cnt_max = 2 ** self.depth - 1
            #: Half-range decision threshold and reset value for the counter.
            self.cnt_half = 2 ** (self.depth - 1)
            #: Non-scaled bipolar counter, expanded to the output shape on use.
            self.cnt: torch.Tensor
            self.register_buffer('cnt', torch.full((1,), float(self.cnt_half), dtype=self.ntype))

        self.encoding_io = {'input_spike': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_spike': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset the local non-scaled counter.

        When **scaled** is ``False``, the counter returns to half of its configured
        range. The cached weight probability is dropped. The threshold sequences
        are unchanged, and ``reset()`` separately restarts the registered encoders
        that hold the sequence position.
        """
        self._weight_prob_cache = None
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
        counter. The threshold sequences are read-only.
        """
        idx = (self.timestep_cur - 1) % self.w_len
        xf = input_spike.type(self.ntype)
        # This is encode's comparison inlined: the bipolar probability map and a
        # strict gt against the sequence entry for this timestep. It stays inline
        # because each input feature carries its own sequence.
        w_spike = torch.gt(self._weight_prob().t(), self.w_num_seq[idx].unsqueeze(-1)).type(self.ntype)
        pc = torch.matmul(xf, w_spike)
        if self.polarity == 'bipolar':
            # sum((1-x)(1-w)) = in_features - sum(x) - sum(w) + sum(xw).
            pc = 2 * pc - xf.sum(-1, keepdim=True) + (self.in_features - w_spike.sum(0))
        if self.has_bias:
            # Bias contributes to the direct path only.
            pc = pc + self.b_encoder(self.bias).type(self.ntype)
        if self.scaled:
            # Strict comparison through the encoder; see the class docstring for
            # the per-count spike loss this replaces the former ge form with.
            return self.reference_encode(pc.div(self.scale_len)).type(self.stype)
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


    def _weight_prob(self):
        """Return the encoder probability view of the weight matrix.

        The value depends only on the weights, so it is cached and rebuilt when
        the weight changes in value, identity, device, or dtype. ``forward()``
        compares it with a hard threshold, which passes no gradient to the
        weights, so the cached value is detached.

        Returns:
            Weight probabilities shaped like ``weight``, in the numeric dtype.
        """
        weight = self.weight
        cache = self._weight_prob_cache
        if (cache is not None and cache[0] is weight and cache[1] == weight._version
                and cache[2].device == weight.device and cache[2].dtype == self.ntype):
            return cache[2]
        with torch.no_grad():
            prob = ((weight + 1) / 2 if self.polarity == 'bipolar' else weight).type(self.ntype)
        self._weight_prob_cache = (weight, weight._version, prob)
        return prob

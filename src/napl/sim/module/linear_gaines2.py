import torch
import math

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import encode, gen_num_seq
from napl.sim.module._shared import _gaines_counter_step
from loguru import logger


class linear_gaines2(napl_base):
    r"""Apply a streaming Gaines layer with per-column number sequences.

    Use this ``gMUL + gADD`` variant to reproduce the UnarySim ``GainesLinear4``
    design, which is written around LFSR sequences. Every input feature carries
    its own rate-coded weight sequence, taken from a distinct Sobol dimension or
    LFSR seed, and a Gaines adder reduces the per-timestep parallel count to one
    output spike. The target is the affine map

    .. math::

       y = Wx + b

    up to the scaling of the selected adder mode, with
    ``entry = in_features + has_bias``:

    - ``scaled=True``: random-threshold addition over a period of
      ``L = 2 ** round(log2(entry))``. Under the default Sobol generator the
      decoded bipolar output represents ``(entry + W x + b) / L - 1``, which is
      ``(W x + b) / entry`` when ``entry`` is a power of two; under LFSR it sits
      about ``1 / entry`` below that, an offset that is negligible at a realistic
      fan-in but reaches a third of full scale at ``entry = 4``.
    - ``scaled=False``: unipolar mode emits ``count > 0``; bipolar mode
      integrates the count in a ``depth``-bit saturating counter and emits
      ``counter > half``.

    Weights are rate-coded from the full-precision tensor (the upstream
    quantizes them to ``bitwidth`` bits first; agreement is within the SC
    bound). Construction stores one threshold sequence per input feature, so no
    stored table grows with the number of output features.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_gaines2

        layer = linear_gaines2(torch.zeros(3, 2),
                               config={"polarity": "bipolar", "timestep": 4,
                                       "generator": "sobol", "scaled": True})
        output_spike = layer(torch.ones(1, 2))

    .. container:: api-references

        .. rubric:: References

        *Stochastic Computing Systems*, Advances in Information Systems Science, 1969.
    """
    #: One threshold sequence per input feature is held inside the layer, so the
    #: RTL counterpart generates the weight and bias streams itself from held
    #: numeric codes instead of taking them as spikes from a shared encoder.
    internal_encode = True


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

        .. container:: api-parameter-list

            **Parameters:**

            - **weight** – Numeric tensor shaped ``(out_features, in_features)``.
            - **bias** – Optional numeric tensor shaped ``(out_features,)``; the default is ``None``.
            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **timestep**: Weight-sequence stream length; the default is ``256``.
              - **generator**: Number-sequence generator name; the default is ``"sobol"``.
              - **dim**: First sequence dimension, with one dimension per input feature above it; the default is ``2``.
              - **seed**: First LFSR seed, advanced per input feature; the default is ``1``.
              - **scaled**: Use random-threshold scaled addition when ``True``; the default is ``True``.
              - **depth**: Bit width of the non-scaled bipolar counter; the default is ``8``.
              - **name**: Optional instance label.

        Scaled mode requires ``in_features + has_bias >= 2``.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['dim', 'scaled', 'depth', 'seed'], polarity_required=True)

        if weight.dim() != 2:
            message = f'linear_gaines2 weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.'
            logger.error(message)
            raise AssertionError(message)
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
            if self.entry < 2:
                message = f'linear_gaines2 scaled mode needs entry >= 2, got {self.entry}.'
                logger.error(message)
                raise AssertionError(message)
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

        # Spike generation and the Gaines adder are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming layer.
        self.hw = hw_params(pp_delay=0)

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
            # The encoder compares the scaled count strictly against its sequence entry.
            return self.reference_encode(pc.div(self.scale_len)).type(self.stype)
        if self.polarity == 'unipolar':
            return torch.gt(pc, 0).type(self.stype)
        # Non-scaled bipolar mode integrates 2*pc-entry around half range.
        delta = 2 * pc - self.entry
        return _gaines_counter_step(self.cnt, delta, self.cnt_max, self.cnt_half).type(self.stype)


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

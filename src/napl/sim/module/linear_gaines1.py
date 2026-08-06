import torch
import math

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import encode, gen_num_seq
from napl.sim.module._shared import _gaines_counter_step
from loguru import logger


class linear_gaines1(napl_base):
    r"""Apply a streaming Gaines ``gMUL + gADD`` fully connected layer.

    Use this variant to reproduce the first UnarySim Gaines linear design or to
    compare random-threshold and counter-based addition. Every input feature
    carries its own rate-coded weight sequence, taken from a distinct Sobol
    dimension, and a Gaines adder replaces :class:`linear`'s scaled accumulator,
    so the target is the affine map

    .. math::

       y = Wx + b

    up to the scaling of the selected adder mode, with
    ``entry = in_features + has_bias``:

    - ``scaled=True``: random-threshold addition over a period of
      ``L = 2 ** round(log2(entry))``. The decoded output represents
      ``(W x + b) / L`` for unipolar streams and
      ``(entry + W x + b) / L - 1`` for bipolar streams.
    - ``scaled=False``: unipolar mode emits ``count > 0``; bipolar mode
      integrates the count in a ``depth``-bit saturating counter, so the decoded
      output tracks ``clamp(W x + b, -1, 1)``.

    This class matches UnarySim ``GainesLinear1``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_gaines1

        layer = linear_gaines1(torch.zeros(3, 2),
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
                'scaled': True,
                'depth': 8,
            }
        ):
        """Construct the Gaines layer from external numeric parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **weight** – Numeric tensor shaped ``(out_features, in_features)``.
            - **bias** – Optional numeric tensor shaped ``(out_features,)``; the default is ``None``.
            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **timestep**: Weight-encoder stream length; the default is ``256``.
              - **generator**: Number-sequence generator name; the default is ``"sobol"``.
              - **dim**: First weight sequence dimension, with one dimension per input feature above it, the bias on ``dim + in_features`` and the scaled threshold on ``dim + in_features + 1``; the default is ``2``.
              - **scaled**: Use random-threshold scaled addition when ``True``; the default is ``True``.
              - **depth**: Bit width of the non-scaled bipolar counter; the default is ``8``.
              - **name**: Optional instance label.

        In scaled mode, the threshold period is
        ``2 ** round(log2(in_features + has_bias))`` and ``in_features +
        has_bias >= 2`` is required.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['dim', 'scaled', 'depth'], polarity_required=True)

        if weight.dim() != 2:
            message = f'linear_gaines1 weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.'
            logger.error(message)
            raise AssertionError(message)
        #: Trainable numeric weight matrix encoded into a spike stream.
        self.weight = torch.nn.Parameter(weight)
        #: Optional trainable numeric bias encoded on its own sequence.
        self.bias = torch.nn.Parameter(bias) if bias is not None else None
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
        self._weight_prob_cache = None

        dim = config.get('dim', 2)
        # Only Sobol-family generators decorrelate input and weight streams by dimension.
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'linear_gaines1 decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim (identical sequences across '
                f'operands). Use a sobol-family generator, or decorrelate the input and weight '
                f'streams by distinct seeds.')
        # Each input column needs a distinct sequence to avoid comonotone weight bits.
        width = math.ceil(math.log2(config['timestep']))
        #: Number of timesteps in the periodic weight spike stream.
        self.w_len = 2 ** width
        if config['generator'].lower() in ['sobol', 'rc', 'rate']:
            # A Sobol column depends only on its own dimension index, so one engine
            # spanning dim .. dim + in_features - 1 supplies every feature sequence.
            num_seq = torch.quasirandom.SobolEngine(dim + self.in_features - 1) \
                .draw(2 ** width)[:, dim - 1:dim - 1 + self.in_features].type(self.ntype)
        else:
            num_seq = torch.stack(
                [gen_num_seq({'width': width, 'generator': config['generator'], 'dim': dim + j})
                 for j in range(self.in_features)], dim=1)
        #: Per-input-feature threshold sequences indexed by timestep.
        self.w_num_seq: torch.Tensor
        self.register_buffer('w_num_seq', num_seq.to(weight.device))
        distinct = torch.unique(self.w_num_seq, dim=1).shape[1]
        if distinct < self.in_features:
            logger.warning(
                f'linear_gaines1 derived only {distinct} distinct weight sequences for '
                f'{self.in_features} input features, so some features share one sequence and '
                f'their weight bits are comonotone. With generator <{config["generator"]}> at '
                f'timestep {config["timestep"]} the sequence period is {self.w_len}; reduce '
                f'in_features below that period or raise timestep.')
        if self.has_bias:
            #: Encoder that converts the optional numeric bias to spikes.
            self.b_encoder = encode({'polarity': self.polarity, 'timestep': config['timestep'],
                                      'generator': config['generator'],
                                      'dim': dim + self.in_features})

        if self.scaled:
            if self.entry < 2:
                message = f'linear_gaines1 scaled mode needs entry >= 2, got {self.entry}.'
                logger.error(message)
                raise AssertionError(message)
            # A full count of 2**w always passes the [0, 2**w) threshold.
            #: Bit width of the scaled-adder threshold sequence.
            self.scale_width = round(math.log2(self.entry))
            #: Number of entries in the scaled-adder threshold sequence.
            self.scale_len = 2 ** self.scale_width
            #: Encoder supplying the scaled-adder threshold comparison.
            self.reference_encode = encode({'polarity': 'unipolar',
                                            'timestep': self.scale_len,
                                            'generator': config['generator'],
                                            'dim': dim + self.in_features + 1})
            scaled_levels = self.reference_encode.num_seq.mul(self.scale_len)
            assert torch.allclose(scaled_levels, scaled_levels.round(), atol=1e-9), \
                f'Sequence value off the 1/{self.scale_len} grid; the count-scale view would not be exact.'
        else:
            depth = config.get('depth', 8)
            #: Maximum value of the non-scaled bipolar saturating counter.
            self.cnt_max = 2 ** depth - 1
            #: Half-range decision threshold and reset value for the counter.
            self.cnt_half = 2 ** (depth - 1)
            if self.polarity == 'bipolar':
                #: Non-scaled bipolar accumulator, expanded to the output shape on use.
                self.cnt: torch.Tensor
                self.register_buffer(
                    'cnt', torch.zeros(1, dtype=self.ntype).fill_(self.cnt_half))

        # Encoders and the Gaines adder are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming layer.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input_spike': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_spike': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset the local non-scaled bipolar counter.

        When **scaled** is ``False`` and polarity is ``"bipolar"``, the counter
        returns to half of its configured range. The cached weight probability is
        dropped. Other configurations have no direct local state to reset.
        """
        self._weight_prob_cache = None
        if not self.scaled and self.polarity == 'bipolar':
            self.cnt.resize_(1).fill_(self.cnt_half)


    def forward(self, input_spike):
        """Process one input-spike timestep.

        Args:
            input_spike: ``0``/``1`` tensor whose last dimension is
                ``in_features``.

        Returns:
            Output spike tensor with last dimension ``out_features``.

        The call advances the optional bias encoder, updates the local counter in
        non-scaled bipolar mode, and advances ``timestep_cur``. The weight
        threshold sequences are read-only.
        """
        idx = (self.timestep_cur - 1) % self.w_len
        xf = input_spike.type(self.ntype)
        # This is encode's comparison inlined: the bipolar probability map and a
        # strict gt against the sequence entry for this timestep. It stays inline
        # because each input feature carries its own sequence.
        w_spike = torch.gt(self._weight_prob().t(),
                           self.w_num_seq[idx].unsqueeze(-1)).type(self.ntype)
        pc = torch.matmul(xf, w_spike)
        # pc is a fresh ntype tensor, so in-place count updates are alias-safe.
        if self.polarity == 'bipolar':
            # sum((1-x)(1-w)) = in_features - sum(x) - sum(w) + sum(xw).
            pc.mul_(2).sub_(xf.sum(-1, keepdim=True)).sub_(w_spike.sum(0)).add_(self.in_features)
        if self.has_bias:
            pc.add_(self.b_encoder(self.bias).type(self.ntype))

        if self.scaled:
            # The encoder compares the scaled count strictly against its sequence entry.
            reference_encode_bit = self.reference_encode(pc.div(self.scale_len))
            output = reference_encode_bit
        elif self.polarity == 'unipolar':
            output = torch.gt(pc, 0)
        else:
            delta = pc.mul_(2).sub_(self.entry)
            output = _gaines_counter_step(self.cnt, delta, self.cnt_max, self.cnt_half)
        return output.type(self.stype)


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

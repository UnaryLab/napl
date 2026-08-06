import torch
import math

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import encode
from napl.sim.module._shared import _gaines_counter_step
from loguru import logger


class linear_gaines1(napl_base):
    r"""Apply a streaming Gaines ``gMUL + gADD`` fully connected layer.

    Use this variant to reproduce the first UnarySim Gaines linear design or to
    compare random-threshold and counter-based addition. Weights are rate-coded
    on a number-sequence dimension separate from the input, and a Gaines adder
    replaces :class:`linear`'s scaled accumulator, so the target is the affine
    map

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
              - **dim**: One-based weight sequence dimension, with the bias on ``dim + 1`` and the scaled threshold on ``dim + 2``; the default is ``2``.
              - **scaled**: Use random-threshold scaled addition when ``True``; the default is ``True``.
              - **depth**: Bit width of the non-scaled bipolar counter; the default is ``8``.
              - **name**: Optional instance label.

        In scaled mode, the threshold period is
        ``2 ** round(log2(in_features + has_bias))``.
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

        dim = config.get('dim', 2)
        # Only Sobol-family generators decorrelate input and weight streams by dimension.
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'linear_gaines1 decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim (identical sequences across '
                f'operands). Use a sobol-family generator, or decorrelate the input and weight '
                f'streams by distinct seeds.')
        #: Encoder that converts the numeric weight matrix to spikes each timestep.
        self.w_encoder = encode({'polarity': self.polarity, 'timestep': config['timestep'],
                                  'generator': config['generator'], 'dim': dim})
        if self.has_bias:
            #: Encoder that converts the optional numeric bias to spikes.
            self.b_encoder = encode({'polarity': self.polarity, 'timestep': config['timestep'],
                                      'generator': config['generator'], 'dim': dim + 1})

        if self.scaled:
            # A full count of 2**w always passes the [0, 2**w) threshold.
            #: Bit width of the scaled-adder threshold sequence.
            self.scale_width = round(math.log2(self.entry))
            #: Number of entries in the scaled-adder threshold sequence.
            self.scale_len = 2 ** self.scale_width
            #: Encoder supplying the scaled-adder threshold comparison.
            self.reference_encode = encode({'polarity': 'unipolar',
                                            'timestep': self.scale_len,
                                            'generator': config['generator'],
                                            'dim': dim + 2})
            scaled_levels = self.reference_encode.num_seq.mul(self.scale_len)
            assert torch.allclose(scaled_levels, scaled_levels.round(), atol=1e-9), \
                f'Sequence value off the 1/{self.scale_len} grid; the count-scale view would not be exact.'
            #: Count-scale view of the encoder sequence, kept for inspection.
            self.scale_seq: torch.Tensor
            self.register_buffer('scale_seq', scaled_levels.round())
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
        returns to half of its configured range. Other configurations have no
        direct local state to reset.
        """
        if not self.scaled and self.polarity == 'bipolar':
            self.cnt.resize_(1).fill_(self.cnt_half)


    def forward(self, input_spike):
        """Process one input-spike timestep.

        Args:
            input_spike: ``0``/``1`` tensor whose last dimension is
                ``in_features``.

        Returns:
            Output spike tensor with last dimension ``out_features``.

        The call advances weight and optional bias encoders, updates the local
        counter in non-scaled bipolar mode, and advances ``timestep_cur``.
        """
        w_spike = self.w_encoder(self.weight)
        xf = input_spike.type(self.ntype)
        wf = w_spike.type(self.ntype)
        pc = torch.matmul(xf, wf.t())
        # pc is a fresh ntype tensor, so in-place count updates are alias-safe.
        if self.polarity == 'bipolar':
            pc.mul_(2).sub_(xf.sum(-1, keepdim=True)).sub_(wf.sum(-1)).add_(self.in_features)
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

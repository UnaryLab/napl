import math

import torch

from napl.sim.base import napl_base
from napl.sim.operation import add_gaines, encode, gen_num_seq
from loguru import logger


class linear_gaines(napl_base):
    r"""Apply a streaming Gaines ``gMUL + gADD`` fully connected layer.

    Use this variant to reproduce the first UnarySim Gaines linear design or to
    compare scaled MUX-select and non-scaled unipolar OR addition. Every
    input feature carries its own rate-coded weight sequence, taken from a distinct
    Sobol dimension, and a Gaines adder replaces :class:`linear_mix`'s scaled accumulator,
    so the target is the affine map

    .. math::

       y = Wx + b

    up to the scaling of the selected adder mode, with
    ``entry = in_features + has_bias``:

    - ``scaled=True``: MUX-based scaled addition over ``entry`` inputs
      (``entry`` must be a power of two). The decoded output represents
      ``(W x + b) / L`` for unipolar streams and
      ``(entry + W x + b) / L - 1`` for bipolar streams.
    - ``scaled=False``: unipolar mode OR-reduces the product bits, approaching
      ``min(1, sum)``. Non-scaled bipolar addition is not supported.

    This class matches UnarySim ``GainesLinear1``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.module import linear_gaines

        layer = linear_gaines(torch.zeros(3, 2),
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
              - **dim**: First weight sequence dimension, with one dimension per input feature above it, the bias on ``dim + in_features`` and the scaled adder's MUX select sequence on ``dim + in_features + 1``; the default is ``2``.
              - **scaled**: Use MUX-select scaled addition, which picks one input per timestep by a Sobol-derived index, when ``True``; the default is ``True``.
              - **name**: Optional instance label.

        In scaled mode, ``in_features + has_bias`` must be a power of two.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['dim', 'scaled'], polarity_required=True)

        #: Whether the Gaines adder uses MUX-select scaled addition, selecting one
        #: input bit per timestep by a Sobol-derived index.
        self.scaled = config.get('scaled', True)
        if self.polarity == 'bipolar' and not self.scaled:
            message = 'Non-scaled Gaines addition in linear_gaines does not support bipolar data.'
            logger.error(message)
            raise AssertionError(message)

        if weight.dim() != 2:
            message = f'linear_gaines weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.'
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
        dim = config.get('dim', 2)
        # Only Sobol-family generators decorrelate input and weight streams by dimension.
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'linear_gaines decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim (identical sequences across '
                f'operands). Use a sobol-family generator, or decorrelate the input and weight '
                f'streams by distinct seeds.')
        # One gen_num_seq per input feature is stacked here so forward thresholds the whole stack in a single torch.gt, standing in for encode at its ceil(log2(timestep)) period width without building encode module objects.
        seq_width = math.ceil(math.log2(config['timestep']))
        w_num_seq = torch.stack([
            gen_num_seq(config={
                'width': seq_width,
                'generator': config['generator'],
                'dim': dim + feature,
            })
            for feature in range(self.in_features)
        ], dim=1)
        #: Encode-owned threshold sequences, one column per input feature.
        self.w_num_seq: torch.Tensor
        self.register_buffer(
            'w_num_seq',
            w_num_seq.to(weight.device),
        )
        distinct = torch.unique(self.w_num_seq, dim=1).shape[1]
        if distinct < self.in_features:
            sequence_period = self.w_num_seq.size(0)
            logger.warning(
                f'linear_gaines derived only {distinct} distinct weight sequences for '
                f'{self.in_features} input features, so some features share one sequence and '
                f'their weight bits are comonotone. With generator <{config["generator"]}> at '
                f'timestep {config["timestep"]} the sequence period is {sequence_period}; reduce '
                f'in_features below that period or raise timestep.')
        if self.has_bias:
            #: Encoder that converts the optional numeric bias to spikes.
            self.b_encoder = encode({
                'polarity': self.polarity,
                'timestep': config['timestep'],
                'generator': config['generator'],
                'dim': dim + self.in_features,
            })

        if self.scaled:
            if self.entry < 2:
                message = f'linear_gaines scaled mode needs entry >= 2, got {self.entry}.'
                logger.error(message)
                raise AssertionError(message)
            if self.entry & (self.entry - 1):
                message = f'linear_gaines scaled mode needs power-of-two entry, got {self.entry}.'
                logger.error(message)
                raise AssertionError(message)
            #: Scaled Gaines adder implemented as a MUX selector over input bits.
            self.acc = add_gaines({'polarity': self.polarity,
                                     'scaled': True,
                                     'entry': self.entry,
                                     'generator': config['generator'],
                                     'dim': dim + self.in_features + 1})
        else:
            #: Non-scaled unipolar Gaines adder reducing the fan-in with an OR.
            self.acc = add_gaines({'polarity': 'unipolar', 'scaled': False})

        # Encoders and the Gaines adder are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming layer.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """Reset no local state after the inherited child-module reset."""
        pass


    def forward(self, input):
        """Process one input-spike timestep.

        Args:
            input: ``0``/``1`` tensor whose last dimension is
                ``in_features``.

        Returns:
            Output spike tensor with last dimension ``out_features``.

        The call advances the optional bias encoder, Gaines adder, and
        ``timestep_cur``. The weight sequence row follows ``timestep_cur``.
        """
        xf = input.type(self.ntype)
        weight_prob = (self.weight + 1) / 2 if self.polarity == 'bipolar' else self.weight
        sequence_index = (self.timestep_cur - 1) % self.w_num_seq.size(0)
        # Inline threshold standing in for encode, which offers no per-feature sequence stack.
        w_bit = torch.gt(
            weight_prob, self.w_num_seq[sequence_index]
        ).type(self.ntype)
        x2d = xf.unsqueeze(-2)
        if self.polarity == 'bipolar':
            # Bipolar multiplication is XNOR on unipolar-encoded bits.
            prod_bits = torch.logical_not(
                torch.logical_xor(x2d.bool(), w_bit.bool())
            ).type(self.ntype)
        else:
            prod_bits = x2d * w_bit

        if self.has_bias:
            b_encoder_bit = self.b_encoder(self.bias).type(self.ntype)
            prefix = [1] * (prod_bits.dim() - 2)
            b_encoder_bit = b_encoder_bit.view(
                *prefix, self.out_features, 1
            ).expand(*prod_bits.shape[:-2], self.out_features, 1)
            add_inputs = torch.cat((prod_bits, b_encoder_bit), dim=-1)
        else:
            add_inputs = prod_bits

        output = self.acc(add_inputs, dim=-1)
        return output.type(self.stype)

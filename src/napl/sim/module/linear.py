import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import add_any
from napl.sim.operation.encode import encode
from loguru import logger


class linear(napl_base):
    r"""Apply a rate-coded unary fully connected layer one timestep at a time.

    Use this layer when inputs are already spike tensors and weights should be
    encoded on a separate number-sequence dimension. It computes the scaled
    affine map one timestep at a time, over unipolar or bipolar rate-coded
    streams,

    .. math::

       y = \frac{Wx + b}{s},

    with **scale** :math:`s` defaulting to ``in_features + has_bias``. The
    output rate reaches that target within the stochastic-computing error of the
    encoded operand streams.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear

        layer = linear(torch.zeros(3, 2), config={"polarity": "bipolar",
                       "timestep": 4, "generator": "sobol"})
        output_spike = layer(torch.ones(1, 2))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
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
                'scale': None,
                'width': 12,
            }
        ):
        """Construct the streaming layer from external numeric parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **weight** – Numeric tensor shaped ``(out_features, in_features)``.
            - **bias** – Optional numeric tensor shaped ``(out_features,)``; the default is ``None``.
            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **timestep**: Weight-encoder stream length; the default is ``256``.
              - **generator**: Number-sequence generator name; the default is ``"sobol"``.
              - **dim**: One-based weight Sobol dimension, with the bias on the next dimension; the default is ``2``.
              - **scale**: Output scaling divisor, where ``None`` uses ``in_features + has_bias``; the default is ``None``.
              - **width**: Signed accumulator width, which must satisfy ``2 ** (width - 1) > in_features + has_bias``; the default is ``12``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['dim', 'scale', 'width'], polarity_required=True)

        if weight.dim() != 2:
            message = f'linear weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.'
            logger.error(message)
            raise AssertionError(message)
        #: Trainable numeric weight encoded into a spike stream.
        self.weight = torch.nn.Parameter(weight)
        #: Optional trainable numeric bias encoded on its own sequence.
        self.bias = torch.nn.Parameter(bias) if bias is not None else None
        #: Number of features produced by the layer.
        self.out_features = weight.shape[0]
        #: Number of features consumed by the layer.
        self.in_features = weight.shape[1]
        #: Whether an encoded bias contributes to each output sum.
        self.has_bias = bias is not None
        #: Unary-adder fan-in, including the bias when present.
        self.entry = self.in_features + (1 if self.has_bias else 0)
        scale = config.get('scale', None)
        #: Divisor implemented by the streaming unary adder.
        self.scale = self.entry if scale is None else scale

        #: Signed accumulator width used by the streaming unary adder.
        self.width = config.get('width', 12)
        # The signed accumulator range must contain every per-step partial sum.
        if 2 ** (self.width - 1) <= self.entry:
            message = (
                f'linear accumulator width <{self.width}> too small for fan-in <{self.entry}>: '
                f'2**(width-1) must be > entry or partial sums saturate. Increase width.'
            )
            logger.error(message)
            raise AssertionError(message)

        dim = config.get('dim', 2)
        # Only Sobol-family generators decorrelate input and weight streams by dimension.
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'linear decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim (identical sequences across '
                f'operands). Use a sobol-family generator, or decorrelate the input and weight '
                f'streams by distinct seeds.')
        #: Encoder that converts the numeric weight to spikes.
        self.w_encoder = encode({'polarity': self.polarity, 'timestep': config['timestep'],
                                  'generator': config['generator'], 'dim': dim})
        #: Streaming unary adder that reduces each linear product count.
        self.acc = add_any({'polarity': self.polarity, 'scale': self.scale, 'width': self.width})

        if self.has_bias:
            #: Encoder that converts the optional numeric bias to spikes.
            self.b_encoder = encode({'polarity': self.polarity, 'timestep': config['timestep'],
                                      'generator': config['generator'], 'dim': dim + 1})

        # Encoders and the unary adder are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming layer.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input_spike': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_spike': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset state owned directly by the layer.

        This class has no extra local state. The inherited ``reset()`` method
        resets the weight and bias encoders and the unary adder.
        """
        pass


    def forward(self, input_spike):
        """Process one input-spike timestep.

        Args:
            input_spike: ``0``/``1`` tensor whose last dimension is
                ``in_features``. Leading dimensions are preserved.

        Returns:
            Output spike tensor with the last dimension replaced by
            ``out_features``.

        The call advances this layer and its registered streaming children and
        updates the adder accumulator. External weight and bias tensors are not
        modified.
        """
        w_spike = self.w_encoder(self.weight)
        xf = input_spike.type(self.ntype)
        wf = w_spike.type(self.ntype)
        psum = torch.matmul(xf, wf.t())
        if self.polarity == 'bipolar':
            # Bipolar XNOR count is 2*sum(xw) - sum(x) - sum(w) + in_features.
            psum = 2 * psum - xf.sum(-1, keepdim=True) - wf.sum(-1) + self.in_features
        if self.has_bias:
            psum = psum + self.b_encoder(self.bias).type(self.ntype)
        return self.acc(psum, entry=self.entry, dim=None)

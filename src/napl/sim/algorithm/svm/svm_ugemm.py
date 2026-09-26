import torch
from loguru import logger

from napl.sim.base import napl_base
from napl.sim.module import linear_ugemm
from napl.sim.operation import encode, gt


class svm_ugemm(napl_base):
    r"""Apply a pre-trained binary linear SVM to a spike stream.

    Each call consumes one bipolar input-spike timestep and returns the scaled
    affine score stream and its unipolar class-decision stream. For weight
    :math:`w`, bias :math:`b`, and linear-layer scale :math:`s`, the streams
    target

    .. math::

       \hat z = (w x + b) / s,\qquad y = \mathbf{1}\{\hat z > 0\}.

    The zero threshold is an independently encoded bipolar rate-0.5 stream.
    This class applies fixed parameters only; it does not train the SVM.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.algorithm.svm import svm_ugemm

        weight = torch.tensor([[0.75, -0.5]])
        bias = torch.tensor([0.125])
        config = {
            'polarity': 'bipolar', 'timestep': 256,
            'generator': 'sobol', 'dim': 2,
        }
        classifier = svm_ugemm(weight, bias, config)
        score_spike, decision_spike = classifier(torch.ones(1, 2))

    .. container:: api-references

        .. rubric:: References

        *Support-Vector Networks*, Machine Learning, 1995.
    """
    #: Whether each call consumes one input-stream timestep.
    streaming = True
    #: The held linear layer and zero-reference source encode numeric operands.
    internal_encode = 'private'


    def __init__(
            self,
            weight,
            bias,
            config={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
                'dim': 2,
                'scale': None,
                'width': 8,
            },
        ):
        """Configure the fixed affine score and zero-reference comparator.

        Args:
            weight: Numeric tensor shaped ``(1, in_features)``.
            bias: Numeric tensor shaped ``(1,)``.
            config: Configuration mapping with required ``polarity``,
                ``timestep``, and ``generator`` keys. Optional ``dim`` selects
                the linear-layer Sobol dimension, ``scale`` selects its output
                divisor, and ``width`` selects its accumulator width.

        Only bipolar streams are supported. The zero-reference encoder uses
        the configured ``dim + 1`` (default ``3``). When **dim** is omitted,
        the score layer uses its own default dimension ``1`` while the
        zero-reference encoder still uses dimension ``3``.
        """
        super().__init__(
            config,
            ['polarity', 'timestep', 'generator'],
            optional_key_list=['dim', 'scale', 'width'],
            polarity_required=True,
        )
        if self.polarity != 'bipolar':
            message = (
                f"Invalid polarity: <{self.polarity}>; legal values: <['bipolar']>."
            )
            logger.error(message)
            raise AssertionError(message)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2 \
                or weight.shape[0] != 1 or weight.shape[1] == 0:
            shape = tuple(weight.shape) if isinstance(weight, torch.Tensor) \
                else type(weight).__name__
            message = (
                f'Invalid weight: <{shape}>; legal values: a non-empty 2-D '
                'tensor shaped (1, in_features).'
            )
            logger.error(message)
            raise AssertionError(message)
        if not isinstance(bias, torch.Tensor) or bias.shape != (1,):
            shape = tuple(bias.shape) if isinstance(bias, torch.Tensor) \
                else type(bias).__name__
            message = (
                f'Invalid bias: <{shape}>; legal values: a 1-D tensor shaped (1,).'
            )
            logger.error(message)
            raise AssertionError(message)

        dim = config.get('dim', 2)
        #: Streaming affine layer that produces the scaled SVM score.
        self.score_layer = linear_ugemm(weight, bias, dict(config))
        #: Encoder for the bipolar-zero comparison reference.
        self.reference_encode = encode({
            'polarity': 'bipolar',
            'timestep': config['timestep'],
            'generator': config['generator'],
            'dim': dim + 1,
        })
        #: Running-rate comparator that emits the class decision.
        self.decision_compare = gt({'polarity': 'bipolar'})
        #: Numeric bipolar zero passed to :attr:`reference_encode` each timestep.
        self.register_buffer('zero', torch.tensor(0.0, dtype=self.ntype))

        self.hw.pp_delay = self.decision_compare.hw.pp_delay
        self.encoding_io = {'input': 'rc', 'score': 'rc', 'decision': 'rc'}
        self.polarity_io = {
            'input': 'bipolar',
            'score': 'bipolar',
            'decision': 'unipolar',
        }
        self.correlation_i = {}


    def _reset(self):
        """Reset class-local state.

        The inherited reset restarts the held score layer, reference encoder,
        and comparator. The fixed zero reference has no mutable state.
        """
        pass


    def forward(self, input):
        """Classify one bipolar input-spike timestep.

        Args:
            input: A 0/1 spike tensor whose last dimension equals the number of
                weight columns.

        Returns:
            A tuple ``(score, decision)``. ``score`` is the bipolar scaled
            affine-score spike, and ``decision`` is the unipolar class spike.
        """
        score = self.score_layer(input)
        reference_encode_bit = self.reference_encode(self.zero)
        decision = self.decision_compare(score, reference_encode_bit)
        return score, decision

import math

import torch
from loguru import logger

from napl.sim.base import napl_base
from napl.sim.operation import (
    add_scale,
    bi2uni,
    encode,
    exp_n1,
    gt,
    mul_scale,
    mul_unibi,
    pow_delay,
    sub_scale,
)


class svm_rbf_ugemm(napl_base):
    r"""Apply a pre-trained RBF-kernel SVM to a spike stream.

    Each call consumes one bipolar feature-vector timestep and returns a
    bipolar scaled-score stream and a unipolar decision stream. For support
    vectors :math:`s_i`, signed dual coefficients :math:`\beta_i`, bias
    :math:`b`, and score scale :math:`q`, the streams target

    .. math::

       \hat f(x) = \frac{\sum_i \beta_i
       \exp(-\gamma_{\rm eff}\lVert x-s_i\rVert^2)+b}{q},\qquad
       y = \mathbf{1}\{\hat f(x)>0\}.

    Each signed difference is divided by 2, squared against a delayed copy,
    converted from bipolar to unipolar, and reduced by the feature count.
    Therefore the norm stream represents
    :math:`\lVert x-s_i\rVert^2/(4D)`. The kernel multiplier applies
    :math:`4D\gamma`; its fixed-point value defines
    ``effective_gamma = kernel_scale / (4 * in_features)``. This formula and
    ``score_scale`` are the complete end-to-end scale compensation.

    ``4 * in_features * gamma`` must not exceed 1, because ``exp_n1`` accepts
    only a unipolar argument in ``[0, 1]``. ``score_scale`` must be at least
    ``sum(abs(beta)) + abs(bias)``. Each support-vector feature, the RBF
    coefficients, beta, bias, and zero reference use distinct Sobol
    dimensions. ``pow_delay`` with ``n=2`` supplies each difference stream with its own
    delayed copy; its depth is the decorrelation contract for the square.
    The beta stream uses a dimension distinct from every kernel operand, as
    required by ``mul_unibi``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.algorithm import svm_rbf_ugemm

        support = torch.tensor([[-0.5, 0.0], [0.5, 0.0]])
        beta = torch.tensor([-0.75, 0.75])
        classifier = svm_rbf_ugemm(
            support, beta, 0.0, 0.0625,
            {'polarity': 'bipolar', 'timestep': 256,
             'generator': 'sobol', 'dim': 2},
        )
        score_spike, decision_spike = classifier(torch.ones(1, 2))

    .. container:: api-references

        .. rubric:: References

        *Support-Vector Networks*, Machine Learning, 1995.
    """
    #: Whether each call consumes one input-stream timestep.
    streaming = True


    def __init__(
            self,
            support_vectors,
            beta,
            bias,
            gamma,
            config={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
                'dim': 2,
                'score_scale': None,
                'intwidth': 12,
                'fracwidth': 8,
                'square_depth': 1,
                'bi2uni_width': 4,
            },
        ):
        """Configure fixed RBF parameters and the streaming composition.

        Args:
            support_vectors: Finite numeric tensor shaped ``(N, D)`` with
                values in ``[-1, 1]``.
            beta: Finite numeric tensor shaped ``(N,)`` with values in
                ``[-1, 1]``.
            bias: Finite scalar in ``[-1, 1]``.
            gamma: Positive finite RBF coefficient satisfying
                ``4 * D * gamma <= 1``.
            config: Configuration mapping with required ``polarity``,
                ``timestep``, and ``generator`` keys. Optional ``dim`` is the
                first internal Sobol dimension, ``score_scale`` is the score
                divisor, ``intwidth`` and ``fracwidth`` select fixed-point
                accumulators, ``square_depth`` selects the square delay, and
                ``bi2uni_width`` selects the polarity-conversion accumulator.

        For ``D`` input features, internal Sobol streams occupy ``dim`` through
        ``dim + D + 6``. Support vectors use the first ``D`` dimensions,
        :class:`exp_n1` uses the next four, then beta, bias, and the zero
        reference use the final three dimensions. A caller-owned input encoder
        must stay outside this complete span.

        Only bipolar input and score streams are supported. The supplied
        parameter tensors are updatable in place; reset restarts stream state.
        """
        super().__init__(
            config,
            ['polarity', 'timestep', 'generator'],
            optional_key_list=[
                'dim', 'score_scale', 'intwidth', 'fracwidth',
                'square_depth', 'bi2uni_width',
            ],
            polarity_required=True,
        )
        if self.polarity != 'bipolar':
            message = (
                f"Invalid polarity: <{self.polarity}>; legal values: <['bipolar']>."
            )
            logger.error(message)
            raise AssertionError(message)
        if not isinstance(support_vectors, torch.Tensor) \
                or support_vectors.ndim != 2 \
                or support_vectors.shape[0] == 0 \
                or support_vectors.shape[1] == 0:
            shape = tuple(support_vectors.shape) \
                if isinstance(support_vectors, torch.Tensor) \
                else type(support_vectors).__name__
            message = (
                f'Invalid support_vectors: <{shape}>; legal values: a non-empty '
                '2-D tensor shaped (support_count, in_features).'
            )
            logger.error(message)
            raise AssertionError(message)
        if not torch.isfinite(support_vectors).all() \
                or torch.any(support_vectors.abs() > 1):
            message = 'Invalid support_vectors: all values must be finite and in [-1, 1].'
            logger.error(message)
            raise AssertionError(message)
        if not isinstance(beta, torch.Tensor) or beta.shape != (support_vectors.shape[0],):
            shape = tuple(beta.shape) if isinstance(beta, torch.Tensor) else type(beta).__name__
            message = (
                f'Invalid beta: <{shape}>; legal values: a 1-D tensor shaped '
                f'({support_vectors.shape[0]},).'
            )
            logger.error(message)
            raise AssertionError(message)
        if not torch.isfinite(beta).all() or torch.any(beta.abs() > 1):
            message = 'Invalid beta: all values must be finite and in [-1, 1].'
            logger.error(message)
            raise AssertionError(message)
        if isinstance(bias, torch.Tensor):
            valid_bias = bias.numel() == 1 and torch.isfinite(bias).all() \
                and torch.all(bias.abs() <= 1)
            bias_value = bias.detach().reshape(()).item() if valid_bias else bias
        else:
            valid_bias = type(bias) in (int, float) and math.isfinite(bias) \
                and abs(bias) <= 1
            bias_value = bias
        if not valid_bias:
            message = f'Invalid bias: <{bias_value}>; legal values: a finite scalar in [-1, 1].'
            logger.error(message)
            raise AssertionError(message)
        if type(gamma) not in (int, float) or gamma <= 0 or not math.isfinite(gamma):
            message = f'Invalid gamma: <{gamma}>; legal values: a positive finite int or float.'
            logger.error(message)
            raise AssertionError(message)

        #: Number of fixed support vectors.
        self.support_count = support_vectors.shape[0]
        #: Number of input features.
        self.in_features = support_vectors.shape[1]
        kernel_scale_requested = 4 * self.in_features * float(gamma)
        if kernel_scale_requested > 1:
            message = (
                f'Invalid gamma: <{gamma}> gives kernel scale <{kernel_scale_requested}>; '
                'legal values satisfy 4 * in_features * gamma <= 1.'
            )
            logger.error(message)
            raise AssertionError(message)

        intwidth = config.get('intwidth', 12)
        fracwidth = config.get('fracwidth', 8)
        square_depth = config.get('square_depth', 1)
        bi2uni_width = config.get('bi2uni_width', 4)
        if type(square_depth) is not int or square_depth < 1:
            message = (
                f'Invalid square_depth: <{square_depth}>; legal values: '
                'an integer of at least 1.'
            )
            logger.error(message)
            raise AssertionError(message)
        if type(bi2uni_width) is not int or bi2uni_width < 2:
            message = (
                f'Invalid bi2uni_width: <{bi2uni_width}>; legal values: '
                'an integer of at least 2.'
            )
            logger.error(message)
            raise AssertionError(message)

        score_bound = beta.abs().sum().item() + abs(float(bias_value))
        requested_score_scale = config.get('score_scale', None)
        if requested_score_scale is None:
            grid = 2.0 ** (-fracwidth)
            requested_score_scale = max(
                grid, math.ceil(score_bound / grid) * grid
            )
        if type(requested_score_scale) not in (int, float) \
                or requested_score_scale <= 0 \
                or not math.isfinite(requested_score_scale) \
                or requested_score_scale < score_bound:
            message = (
                f'Invalid score_scale: <{requested_score_scale}>; legal values: '
                f'a positive finite number at least <{score_bound}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Fixed support vectors encoded on every timestep.
        self.support_vectors = torch.nn.Parameter(support_vectors)
        #: Fixed signed dual coefficients encoded on every timestep.
        self.beta = torch.nn.Parameter(beta)
        #: Fixed affine bias encoded on every timestep.
        self.bias = torch.nn.Parameter(torch.as_tensor(bias_value, dtype=support_vectors.dtype))
        #: Requested RBF coefficient before fixed-point quantization.
        self.gamma = float(gamma)

        dim = config.get('dim', 2)
        codec = {
            'polarity': 'bipolar',
            'timestep': config['timestep'],
            'generator': config['generator'],
        }
        support_encoders = [
            encode({**codec, 'dim': dim + index})
            for index in range(self.in_features)
        ]
        for index, operation in enumerate(support_encoders):
            self.add_module(f'support_encoder_{index}', operation)
        #: One independently encoded support-vector stream per feature.
        self.support_encoders = tuple(support_encoders)
        difference_ops = [
            sub_scale({
                'polarity': 'bipolar', 'scale': 2,
                'intwidth': intwidth, 'fracwidth': fracwidth,
            }) for _ in range(self.in_features)
        ]
        for index, operation in enumerate(difference_ops):
            self.add_module(f'difference_op_{index}', operation)
        #: Scaled signed subtractors, each producing ``(x_d - s_id) / 2``.
        self.difference_ops = tuple(difference_ops)
        square_ops = [
            pow_delay({
                'polarity': 'bipolar', 'n': 2, 'depth': square_depth,
            })
            for _ in range(self.in_features)
        ]
        for index, operation in enumerate(square_ops):
            self.add_module(f'square_op_{index}', operation)
        #: Delayed-copy square operations, one per feature.
        self.square_ops = tuple(square_ops)
        square_converters = [
            bi2uni({'width': bi2uni_width}) for _ in range(self.in_features)
        ]
        for index, operation in enumerate(square_converters):
            self.add_module(f'square_converter_{index}', operation)
        #: Bipolar-to-unipolar converters for the non-negative squares.
        self.square_converters = tuple(square_converters)
        #: Feature reducer producing ``||x - s_i||^2 / (4 * D)``.
        self.norm_add = add_scale({
            'polarity': 'unipolar', 'scale': self.in_features,
            'intwidth': intwidth, 'fracwidth': fracwidth,
        })
        #: Scaler producing the effective ``gamma * ||x - s_i||^2`` argument.
        self.kernel_scale_op = mul_scale({
            'polarity': 'unipolar', 'scale': kernel_scale_requested,
            'intwidth': intwidth, 'fracwidth': fracwidth,
        })
        #: Effective fixed-point kernel multiplier.
        self.kernel_scale = self.kernel_scale_op.scale
        #: Effective RBF coefficient after fixed-point quantization.
        self.effective_gamma = self.kernel_scale / (4 * self.in_features)
        # exp_n1 consumes four consecutive coefficient-stream dimensions, so
        # the next independent stream starts at dim + in_features + 4.
        #: Negative-exponential kernel stage.
        self.kernel_exp = exp_n1({
            'polarity': 'unipolar', 'timestep': config['timestep'],
            'generator': config['generator'], 'dim': dim + self.in_features,
        })
        #: Encoder for the signed dual coefficients.
        self.beta_encoder = encode({
            **codec, 'dim': dim + self.in_features + 4,
        })
        #: Mixed-polarity support-vector product stage.
        self.kernel_beta_mul = mul_unibi()
        #: Encoder for the affine bias.
        self.bias_encoder = encode({
            **codec, 'dim': dim + self.in_features + 5,
        })
        #: Scaled support-vector and bias reducer.
        self.score_add = add_scale({
            'polarity': 'bipolar', 'scale': requested_score_scale,
            'intwidth': intwidth, 'fracwidth': fracwidth,
        })
        if self.score_add.scale < score_bound:
            message = (
                f'Quantized score_scale <{self.score_add.scale}> is below the '
                f'coefficient L1 bound <{score_bound}>; increase score_scale or fracwidth.'
            )
            logger.error(message)
            raise AssertionError(message)
        #: Effective score divisor after fixed-point quantization.
        self.score_scale = self.score_add.scale
        #: Encoder for the bipolar-zero comparison reference.
        self.reference_encode = encode({
            **codec, 'dim': dim + self.in_features + 6,
        })
        #: Running-rate comparator that emits the class decision.
        self.decision_compare = gt({'polarity': 'bipolar'})
        #: Numeric bipolar zero passed to the comparison-reference encoder.
        self.register_buffer('zero', torch.tensor(0.0, dtype=self.ntype))

        #: Encoder the hardware counterpart carries, derived from the registered
        #: parts: ``'private'`` when any part carries an encoder of its own,
        #: ``'none'`` otherwise.
        self.internal_encode = 'private' if any(part.internal_encode != 'none' for part in self.children()) else 'none'
        self.hw.pp_delay = self.decision_compare.hw.pp_delay
        self.encoding_io = {'input': 'rc', 'score': 'rc', 'decision': 'rc'}
        self.polarity_io = {
            'input': 'bipolar', 'score': 'bipolar', 'decision': 'unipolar',
        }
        self.correlation_i = {}


    def _reset(self):
        """Reset no local state beyond the registered child operations."""
        pass


    def forward(self, input):
        """Evaluate one timestep for every input feature vector.

        Args:
            input: A 0/1 spike tensor whose last dimension is ``in_features``.

        Returns:
            ``(score, decision)`` with the input's leading shape. ``score`` is
            bipolar and divided by ``score_scale``; ``decision`` is unipolar.
        """
        squares = []
        for index in range(self.in_features):
            support_bit = self.support_encoders[index](
                self.support_vectors[:, index]
            )
            input_bit, support_bit = torch.broadcast_tensors(
                input[..., index].unsqueeze(-1), support_bit
            )
            difference = self.difference_ops[index](
                input_bit, support_bit
            )
            square = self.square_ops[index](difference)
            squares.append(self.square_converters[index](square))
        norm = self.norm_add(torch.stack(squares, dim=-1), dim=-1)
        kernel = self.kernel_exp(self.kernel_scale_op(norm))
        beta_bit = self.beta_encoder(self.beta)
        terms = self.kernel_beta_mul(kernel, beta_bit)
        bias_bit = self.bias_encoder(self.bias).expand(terms.shape[:-1])
        score = self.score_add(
            torch.cat((terms, bias_bit.unsqueeze(-1)), dim=-1), dim=-1
        )
        reference_encode_bit = self.reference_encode(self.zero)
        decision = self.decision_compare(score, reference_encode_bit)
        if decision.shape != score.shape:
            decision = decision.expand(score.shape)
        return score, decision

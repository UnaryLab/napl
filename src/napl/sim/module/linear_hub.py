import torch
import math

from napl.utils import rshift_offset
from napl.sim.base import napl_base
from loguru import logger
from napl.sim.module._shared import _init_linear_params, _build_hub_map, _linear_hub_fn


class linear_hub(napl_base):
    r"""Apply a hybrid unary-binary approximation of ``torch.nn.Linear``.

    Use this single-shot trainable layer when products should use a precomputed
    unary multiplication value map while the interface remains numeric. Input and weight are
    quantized to sign-magnitude fixed point, and each absolute input-weight product is looked
    up from a precomputed value map that emulates the unary (bitstream-AND) multiplication
    under the chosen RNG. It is single-shot, trains through a straight-through
    estimator, and approximates ``nn.Linear``
    within the unary-multiplication bound. Rate coding, sign-magnitude.
    Requires ``widthi == widthw``.

    The precise target is the affine map

    .. math::

       y = x W^{\top} + b.

    Let :math:`r_i`, :math:`r_w`, and :math:`r_o` be the dynamic shifts from
    ``rshift_offset`` and :math:`C = 2^{\text{width}-1}` the cycle count. The
    value map holds the bitstream AND-count of two unary streams driven by the
    RNG sequences :math:`g^i` and :math:`g^w`,

    .. math::

       M_{ab} = \sum_{k < m_a} \mathbf{1}\{b > g^{w}_{k}\},\qquad
       m_a = \sum_{k < C} \mathbf{1}\{a > g^{i}_{k}\},

    and the layer evaluates

    .. math::

       \hat x = \mathrm{clamp}\left(
       \left\lfloor |x\,2^{-r_i}| \right\rfloor,\, 0,\, C-1\right),\qquad
       \hat W = \mathrm{clamp}\left(
       \left\lfloor |W\,2^{-r_w}| \right\rfloor,\, 0,\, C-1\right),

    .. math::

       y = \left(\mathrm{sgn}(x)\left[
       M_{\hat x \hat W}\,\mathrm{sgn}(W)\right]^{\top}\right) 2^{-r_o} + b.

    Magnitudes are truncated rather than rounded, and each product is the unary
    AND-count instead of an exact product, so the error is bounded by the
    unary-multiplication bound at :math:`C` cycles.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_hub

        layer = linear_hub(2, 3, config={"widthi": 4, "widthw": 4,
                                        "cycle": 8})
        output = layer(torch.zeros(1, 2))
    """
    streaming = False


    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            weight_ext=None,
            bias_ext=None,
            config={
                'widthi': 8,
                'rngi': 'sobol',
                'quantilei': 1,
                'widthw': 8,
                'rngw': 'sobol',
                'quantilew': 1,
                'cycle': 128,
                'rounding': 'round',
            }
        ):
        """Construct the HUB layer and its unary-product lookup map.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Create a trainable bias when ``True``. Defaults to ``True``.
            weight_ext: Optional initial weight tensor. Defaults to ``None``.
            bias_ext: Optional initial bias tensor. Defaults to ``None``.
            config: Configuration mapping with these keys:

                * **widthi**, **widthw** - Equal sign-magnitude widths. Both
                  default to ``8``.
                * **rngi**, **rngw** - Input and weight RNG names. Both default
                  to ``"sobol"``; ``"rc"`` and ``"tc"`` follow the implemented
                  map rules.
                * **quantilei**, **quantilew** - Dynamic-scaling quantiles. Both
                  default to ``1``.
                * **cycle** - Active unary cycles, capped at ``2 ** (widthi - 1)``.
                  The declared default is ``128``; ``None`` selects the cap.
                * **rounding** - Dynamic-scaling rounding mode. Defaults to
                  ``"round"``.
                * **name** - Optional instance label. Defaults to ``None``.

        The value map is persistent non-trainable state; weights and optional bias
        are trainable parameters.
        """
        super().__init__(config, [])
        #: Number of features consumed by the layer.
        self.in_features = in_features
        #: Number of features produced by the layer.
        self.out_features = out_features
        #: Quantization width used for input values.
        self.widthi = config.get('widthi', 8)
        #: Quantization width used for weight values.
        self.widthw = config.get('widthw', 8)
        assert self.widthi == self.widthw, \
            logger.error(f'linear_hub requires widthi == widthw (got {self.widthi}, {self.widthw}).')
        #: Number-sequence generator used for input values.
        self.rngi = config.get('rngi', 'sobol').lower()
        #: Number-sequence generator used for weight values.
        self.rngw = config.get('rngw', 'sobol').lower()
        #: Input-magnitude quantile used to choose the scaling shift.
        self.quantilei = config.get('quantilei', 1)
        #: Weight-magnitude quantile used to choose the scaling shift.
        self.quantilew = config.get('quantilew', 1)
        #: Rounding mode used during quantization.
        self.rounding = config.get('rounding', 'round').lower()
        # Sign-magnitude encoding reserves one bit, so cycle_max is 2**(width-1).
        #: Maximum cycle count supported by the unary product map.
        self.cycle_max, mapcbsg = _build_hub_map(self.widthi, self.widthw, self.rngi, self.rngw, self.ntype)
        cycle_cfg = config.get('cycle', None)
        #: Cycle count used for each unary product.
        self.cycle_act = self.cycle_max if cycle_cfg is None else min(cycle_cfg, self.cycle_max)
        #: Lookup map used to evaluate unary products.
        self.mapcbsg: torch.Tensor
        self.register_buffer('mapcbsg', mapcbsg)

        _init_linear_params(self, in_features, out_features, bias, weight_ext, bias_ext)

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset local execution state.

        This single-shot layer has no mutable run state. The lookup map and
        trainable parameters are unchanged.
        """
        pass


    def forward(self, input):
        """Apply the HUB linear approximation.

        Args:
            input: Two-dimensional numeric tensor shaped
                ``(batch, in_features)``.

        Returns:
            Numeric tensor shaped ``(batch, out_features)``.

        The call reads the lookup map and computes dynamic shifts without changing
        persistent state or ``timestep_cur``. Backpropagation uses a
        straight-through linear gradient.
        """
        rshift_i, rshift_w, rshift_o = rshift_offset(input, self.weight, self.widthi - 1, self.widthw - 1,
                                                     self.rounding, self.quantilei, self.quantilew)
        return _linear_hub_fn.apply(input, self.weight, self.bias,
                                    rshift_i, rshift_w, rshift_o, self.cycle_act, self.mapcbsg)

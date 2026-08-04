import torch
import math

from napl.utils import rshift_offset
from napl.sim.base import napl_base
from loguru import logger
from napl.sim.module._shared import _init_linear_params, _linear_fxp_fn


class linear_fxp(napl_base):
    """Apply a trainable fixed-point approximation of ``torch.nn.Linear``.

    Use this single-shot layer for quantization-aware evaluation or training. It dynamically scales input and weight
    to ``widthi``- and ``widthw``-bit fixed point through ``rshift_offset``,
    multiplies them, then restores the output scale. It is single-shot, trains
    through a straight-through estimator, and approximates ``nn.Linear`` within
    the quantization bound.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_fxp

        layer = linear_fxp(2, 3)
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
                'quantilei': 1,
                'widthw': 8,
                'quantilew': 1,
                'rounding': 'round',
            }
        ):
        """Construct the fixed-point layer and initialize trainable parameters.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Create a trainable bias when ``True``. Defaults to ``True``.
            weight_ext: Optional initial weight tensor shaped
                ``(out_features, in_features)``. Defaults to ``None``.
            bias_ext: Optional initial bias tensor shaped ``(out_features,)``.
                Used only when **bias** is ``True``. Defaults to ``None``.
            config: Configuration mapping with **widthi** and **widthw** (input
                and weight widths, both default ``8``), **quantilei** and
                **quantilew** (scaling quantiles, both default ``1``), and
                **rounding** (rounding mode, default ``"round"``). **name** is an
                optional instance label and defaults to ``None``.
        """
        super().__init__(config, [])
        #: Number of features consumed by the layer.
        self.in_features = in_features
        #: Number of features produced by the layer.
        self.out_features = out_features
        #: Fixed-point width used for input values.
        self.widthi = config.get('widthi', 8)
        #: Fixed-point width used for weight values.
        self.widthw = config.get('widthw', 8)
        #: Input-magnitude quantile used to choose the scaling shift.
        self.quantilei = config.get('quantilei', 1)
        #: Weight-magnitude quantile used to choose the scaling shift.
        self.quantilew = config.get('quantilew', 1)
        #: Rounding mode used during fixed-point conversion.
        self.rounding = config.get('rounding', 'round').lower()
        #: Largest input magnitude represented by the quantized kernel.
        self.max_abs_i = 2 ** (self.widthi - 1)
        #: Largest weight magnitude represented by the quantized kernel.
        self.max_abs_w = 2 ** (self.widthw - 1)
        _init_linear_params(self, in_features, out_features, bias, weight_ext, bias_ext)


    def _reset(self):
        """Reset local execution state.

        This single-shot layer has no mutable run state, so the hook returns
        ``None`` without changing trainable parameters.
        """
        pass


    def forward(self, input):
        """Apply the fixed-point linear approximation.

        Args:
            input: Numeric tensor shaped ``(batch, in_features)``.

        Returns:
            Numeric tensor shaped ``(batch, out_features)``.

        The call computes dynamic shifts from the input and weight. It does not
        change persistent state or ``timestep_cur``; the custom backward uses the
        straight-through linear gradient.
        """
        rshift_i, rshift_w, _ = rshift_offset(input, self.weight, self.widthi - 1, self.widthw - 1,
                                              self.rounding, self.quantilei, self.quantilew)
        rshift_o = 0 - rshift_i - rshift_w
        return _linear_fxp_fn.apply(input, self.weight, self.bias,
                                    rshift_i, rshift_w, rshift_o, self.max_abs_i, self.max_abs_w,
                                    True)

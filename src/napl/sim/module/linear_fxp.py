from napl.utils import rshift_offset
from napl.sim.base import napl_base
from napl.sim.module._shared import _init_linear_params, _linear_fxp_fn

# Single source for every optional key: the signature default and the per-key fallback.
_DEFAULT_CONFIG = {
    'widthi': 8,
    'quantilei': 1,
    'widthw': 8,
    'quantilew': 1,
    'rounding': 'round',
}


class linear_fxp(napl_base):
    r"""Apply a trainable fixed-point approximation of ``torch.nn.Linear``.

    Use this single-shot layer for quantization-aware evaluation or training.
    Input and weight are dynamically scaled to ``widthi``- and ``widthw``-bit
    signed fixed point, and the target is the affine map

    .. math::

       y = x W^{\top} + b.

    The operands are not used exactly: each is rounded and clamped to its signed
    fixed-point range, written :math:`Q_i` and :math:`Q_w`,

    .. math::

       y \approx Q_i(x)\, Q_w(W)^{\top} + b,

    so the error is the fixed-point quantization. It trains through a
    straight-through estimator.

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
            config=_DEFAULT_CONFIG
        ):
        """Construct the fixed-point layer and initialize trainable parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **in_features** – Number of input features.
            - **out_features** – Number of output features.
            - **bias** – Create a trainable bias when ``True``; the default is ``True``.
            - **weight_ext** – Optional initial weight tensor shaped ``(out_features, in_features)``; the default is ``None``.
            - **bias_ext** – Optional initial bias tensor shaped ``(out_features,)``, used only when **bias** is ``True``; the default is ``None``.
            - **config** – Configuration mapping. Omitted keys fall back to the same defaults.

              - **widthi**: Fixed-point width for input values; the default is ``8``.
              - **quantilei**: Input-magnitude scaling quantile; the default is ``1``.
              - **widthw**: Fixed-point width for weight values; the default is ``8``.
              - **quantilew**: Weight-magnitude scaling quantile; the default is ``1``.
              - **rounding**: Dynamic-scaling rounding mode; the default is ``"round"``.
              - **name**: Optional instance label.
        """
        super().__init__(config, [], optional_key_list=list(_DEFAULT_CONFIG))
        cfg = {**_DEFAULT_CONFIG, **config}
        #: Number of features consumed by the layer.
        self.in_features = in_features
        #: Number of features produced by the layer.
        self.out_features = out_features
        #: Fixed-point width used for input values.
        self.widthi = cfg['widthi']
        #: Fixed-point width used for weight values.
        self.widthw = cfg['widthw']
        #: Input-magnitude quantile used to choose the scaling shift.
        self.quantilei = cfg['quantilei']
        #: Weight-magnitude quantile used to choose the scaling shift.
        self.quantilew = cfg['quantilew']
        #: Rounding mode used when choosing the scaling shift.
        self.rounding = cfg['rounding'].lower()
        #: Largest input magnitude represented by the quantized kernel.
        self.max_abs_i = 2 ** (self.widthi - 1)
        #: Largest weight magnitude represented by the quantized kernel.
        self.max_abs_w = 2 ** (self.widthw - 1)
        _init_linear_params(self, in_features, out_features, bias, weight_ext, bias_ext)

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


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

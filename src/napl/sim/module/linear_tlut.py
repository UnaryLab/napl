import torch
import math

from napl.sim.base import napl_base
from loguru import logger
from napl.sim.module._shared import (
    _init_linear_params,
    _linear_tlut_fxpfxp_fn,
    _linear_tlut_fxpfp_fn,
    _linear_tlut_fpfp_fn,
    _TLUT_FP_WIDTH,
)


class linear_tlut(napl_base):
    """Apply a temporal-LUT approximation of ``torch.nn.Linear``.

    Use this single-shot trainable layer to decompose either the input or weight
    into temporal digits while retaining a numeric interface. The chosen operand
    is decomposed into a sum of ``widtht``-bit temporal digits and accumulated,
    while the other operand stays fixed-point or floating-point. The
    ``(formati, formatw)`` pair selects ``fxpfxp``, ``fxpfp``, or ``fpfp`` mode.
    The layer is single-shot, trains through a straight-through estimator, and
    approximates ``nn.Linear`` within the temporal-decomposition bound.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_tlut

        layer = linear_tlut(2, 3, config={"temporal": "i", "widtht": 4,
                                         "formati": "fxp", "widthi": 8,
                                         "formatw": "fxp", "widthw": 8})
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
                'temporal': 'i',
                'widtht': 4,
                'formati': 'fxp',
                'widthi': 8,
                'quantilei': 1,
                'formatw': 'fxp',
                'widthw': 8,
                'quantilew': 1,
                'cycle': None,
                'rounding': 'round',
            }
        ):
        """Construct the temporal-LUT layer and select its execution mode.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Create a trainable bias when ``True``. Defaults to ``True``.
            weight_ext: Optional initial weight tensor. Defaults to ``None``.
            bias_ext: Optional initial bias tensor. Defaults to ``None``.
            config: Configuration mapping with these keys:

                * **temporal** - ``"i"`` or ``"input"`` to decompose inputs;
                  ``"w"`` or ``"weight"`` to decompose weights. Defaults to
                  ``"i"``.
                * **widtht** - Bits per temporal digit. Defaults to ``4``.
                * **formati**, **formatw** - Operand formats. ``"fxp"`` selects
                  fixed point; floating formats must be keys in the implemented
                  width map: ``"bfloat16"``, ``"float16"``, or ``"float32"``.
                  Both default to ``"fxp"``.
                * **widthi**, **widthw** - Fixed-point widths. Both default to ``8``.
                * **quantilei**, **quantilew** - Scaling quantiles. Both default
                  to ``1``.
                * **cycle** - Active cycles, capped at ``2 ** widtht``. ``None``
                  selects the cap and is the default.
                * **rounding** - Fixed-point rounding mode. Defaults to ``"round"``.
                * **name** - Optional instance label. Defaults to ``None``.
        """
        super().__init__(config, [])
        #: Number of features consumed by the layer.
        self.in_features = in_features
        #: Number of features produced by the layer.
        self.out_features = out_features
        #: Operand decomposed into temporal digits.
        self.temporal = config.get('temporal', 'i').lower()
        #: Number of bits represented by each temporal digit.
        self.widtht = config.get('widtht', 4)
        #: Numeric format used for input values.
        self.formati = config.get('formati', 'fxp').lower()
        #: Numeric format used for weight values.
        self.formatw = config.get('formatw', 'fxp').lower()
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
        assert self.temporal in ('i', 'input', 'w', 'weight'), \
            logger.error(f"linear_tlut 'temporal' must be one of ['i','input','w','weight'], got {self.temporal}.")

        if self.formati == 'fxp' and self.formatw == 'fxp':
            #: Arithmetic path selected from the input and weight formats.
            self.mode = 'fxpfxp'
        elif self.formati != 'fxp' and self.formatw != 'fxp':
            #: Arithmetic path selected from the input and weight formats.
            self.mode = 'fpfp'
        else:
            #: Arithmetic path selected from the input and weight formats.
            self.mode = 'fxpfp'

        #: Maximum number of temporal cycles per product.
        self.cycle_max = 2 ** self.widtht
        cycle_cfg = config.get('cycle', None)
        #: Number of temporal cycles used per product.
        self.cycle_act = self.cycle_max if cycle_cfg is None else min(cycle_cfg, self.cycle_max)
        #: Input magnitude width excluding its sign bit.
        self.widthi_mag = self.widthi - 1
        #: Weight magnitude width excluding its sign bit.
        self.widthw_mag = self.widthw - 1

        if self.temporal in ('i', 'input'):
            fmt = self.formati
            #: Bit width of the operand decomposed into temporal digits.
            self.width = self.widthi - 1 if fmt == 'fxp' else _TLUT_FP_WIDTH[fmt]
        else:
            fmt = self.formatw
            #: Bit width of the operand decomposed into temporal digits.
            self.width = self.widthw - 1 if fmt == 'fxp' else _TLUT_FP_WIDTH[fmt]
        #: Number of temporal digits required for the selected operand.
        self.degree = int(math.ceil(self.width / self.widtht))
        #: Leading padding bits in the temporal decomposition.
        self.delta = int(self.degree * self.widtht - self.width)

        _init_linear_params(self, in_features, out_features, bias, weight_ext, bias_ext)


    def _reset(self):
        """Reset local execution state.

        This single-shot layer has no mutable run state, so the hook returns
        ``None`` without changing trainable parameters.
        """
        pass


    def forward(self, input):
        """Apply the selected temporal-LUT linear approximation.

        Args:
            input: Numeric tensor shaped ``(batch, in_features)``.

        Returns:
            Numeric tensor shaped ``(batch, out_features)``.

        The call does not change persistent state or ``timestep_cur``. The
        selected custom autograd path supplies a straight-through linear gradient.
        """
        cp, cn = self.cycle_act, -self.cycle_act
        if self.mode == 'fxpfxp':
            return _linear_tlut_fxpfxp_fn.apply(input, self.weight, self.bias, self.temporal,
                                                self.widthi_mag, self.widthw_mag, self.widtht, self.degree,
                                                self.delta, cp, cn, self.rounding, self.quantilei, self.quantilew)
        elif self.mode == 'fxpfp':
            return _linear_tlut_fxpfp_fn.apply(input, self.weight, self.bias, self.temporal,
                                               self.width, self.widtht, self.degree, self.delta,
                                               cp, cn, self.rounding, self.quantilei, self.quantilew)
        else:
            return _linear_tlut_fpfp_fn.apply(input, self.weight, self.bias, self.temporal,
                                              self.width, self.widtht, self.degree, self.delta, cp, cn)

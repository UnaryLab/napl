import math

from napl.sim.base import napl_base
from napl.sim.module._shared import (
    _init_linear_params,
    _linear_tlut_fxpfxp_fn,
    _linear_tlut_fxpfp_fn,
    _linear_tlut_fpfp_fn,
    _TLUT_FP_WIDTH,
)
from loguru import logger

# Single source for every optional key: the signature default and the per-key fallback.
_DEFAULT_CONFIG = {
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


class linear_tlut(napl_base):
    r"""Apply a temporal-LUT approximation of ``torch.nn.Linear``.

    Use this single-shot trainable layer to decompose either the input or weight
    into temporal digits while retaining a numeric interface. The
    ``(formati, formatw)`` pair selects ``fxpfxp``, ``fxpfp``, or ``fpfp`` mode,
    and the target is the affine map

    .. math::

       y = x W^{\top} + b.

    The operand named by **temporal** is not used exactly: it is truncated to
    fixed point and split into ``degree`` digits of ``widtht`` bits each, with
    digit :math:`f_k` clamped to the active cycle range,

    .. math::

       x \approx \sum_{k=1}^{\text{degree}} f_k\, 2^{-k\,\text{widtht}}.

    The digits recompose the truncated value exactly, so the error is the
    fixed-point truncation plus the clamp, which acts only when **cycle** is
    below its cap. The layer trains through a straight-through estimator.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_tlut

        layer = linear_tlut(2, 3, config={"temporal": "i", "widtht": 4,
                                         "formati": "fxp", "widthi": 8,
                                         "formatw": "fxp", "widthw": 8})
        output = layer(torch.zeros(1, 2))

    .. container:: api-references

        .. rubric:: References

        *Carat: Unlocking Value-Level Parallelism for Multiplier-Free GEMMs*, ASPLOS, 2024.

        *T-MAC: Temporal Multiplication with Accumulation*, Young Architect Workshop, 2022.
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
        """Construct the temporal-LUT layer and select its execution mode.

        .. container:: api-parameter-list

            **Parameters:**

            - **in_features** – Number of input features.
            - **out_features** – Number of output features.
            - **bias** – Create a trainable bias when ``True``; the default is ``True``.
            - **weight_ext** – Optional initial weight tensor; the default is ``None``.
            - **bias_ext** – Optional initial bias tensor; the default is ``None``.
            - **config** – Configuration mapping. Omitted keys fall back to the same defaults.

              - **temporal**: Operand decomposed into temporal digits, ``"i"``/``"input"`` or ``"w"``/``"weight"``; the default is ``"i"``.
              - **widtht**: Bits per temporal digit; the default is ``4``.
              - **formati**: Input format, ``"fxp"`` or one of ``"bfloat16"``, ``"float16"``, ``"float32"``; the default is ``"fxp"``.
              - **widthi**: Fixed-point width for input values; the default is ``8``.
              - **quantilei**: Input-magnitude scaling quantile; the default is ``1``.
              - **formatw**: Weight format, with the same choices as **formati**; the default is ``"fxp"``.
              - **widthw**: Fixed-point width for weight values; the default is ``8``.
              - **quantilew**: Weight-magnitude scaling quantile; the default is ``1``.
              - **cycle**: Active cycles, capped at ``2 ** widtht``, where ``None`` selects the cap; the default is ``None``.
              - **rounding**: Dynamic-scaling rounding mode; the default is ``"round"``.
              - **name**: Optional instance label.
        """
        super().__init__(config, [], optional_key_list=list(_DEFAULT_CONFIG))
        cfg = {**_DEFAULT_CONFIG, **config}
        #: Number of features consumed by the layer.
        self.in_features = in_features
        #: Number of features produced by the layer.
        self.out_features = out_features
        #: Operand decomposed into temporal digits.
        self.temporal = cfg['temporal'].lower()
        #: Number of bits represented by each temporal digit.
        self.widtht = cfg['widtht']
        #: Numeric format used for input values.
        self.formati = cfg['formati'].lower()
        #: Numeric format used for weight values.
        self.formatw = cfg['formatw'].lower()
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
        if self.temporal not in ('i', 'input', 'w', 'weight'):
            message = f"linear_tlut 'temporal' must be one of ['i','input','w','weight'], got {self.temporal}."
            logger.error(message)
            raise AssertionError(message)

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
        cycle_cfg = cfg['cycle']
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

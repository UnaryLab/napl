import torch

from napl.utils import rshift_offset
from napl.sim.base import napl_base
from napl.sim.module._shared import _init_linear_params, _build_hub_map, _check_hub_int, _linear_hub_fn
from loguru import logger

# Single source for every optional key: the signature default and the per-key fallback.
_DEFAULT_CONFIG = {
    'widthi': 8,
    'rngi': 'sobol',
    'quantilei': 1,
    'widthw': 8,
    'rngw': 'sobol',
    'quantilew': 1,
    'cycle': None,
    'rounding': 'round',
}


class linear_hub(napl_base):
    r"""Apply a hybrid unary-binary approximation of ``torch.nn.Linear``.

    Use this single-shot trainable layer when products should use a precomputed
    unary multiplication value map while the interface remains numeric. Input
    and weight are quantized to sign-magnitude fixed point with rate coding, and
    the target is the affine map

    .. math::

       y = x W^{\top} + b.

    Each magnitude product is not evaluated exactly: the magnitudes are quantized
    to integer levels :math:`a` and :math:`b`, which index a value map holding
    the bitstream AND-count of two unary streams over
    :math:`C = 2^{\text{widthi}-1}` cycles,

    .. math::

       \frac{a}{C}\,\frac{b}{C} \approx \frac{\mathrm{map}[a, b]}{C},

    so the error is the level quantization plus the unary-multiplication error at
    :math:`C` cycles. The layer trains through a straight-through estimator and
    requires ``widthi == widthw``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_hub

        layer = linear_hub(2, 3, config={"widthi": 4, "widthw": 4,
                                        "cycle": 8})
        output = layer(torch.zeros(1, 2))

    .. container:: api-references

        .. rubric:: References

        *uSystolic: Byte-Crawling Unary Systolic Array*, HPCA, 2022.
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
        """Construct the HUB layer and its unary-product lookup map.

        .. container:: api-parameter-list

            **Parameters:**

            - **in_features** – Number of input features.
            - **out_features** – Number of output features.
            - **bias** – Create a trainable bias when ``True``; the default is ``True``.
            - **weight_ext** – Optional initial weight tensor; the default is ``None``.
            - **bias_ext** – Optional initial bias tensor; the default is ``None``.
            - **config** – Configuration mapping. Omitted keys fall back to the same defaults.

              - **widthi**: Sign-magnitude width for input values, which must equal **widthw**; the default is ``8``.
              - **rngi**: Input RNG name, one of ``"sobol"``, ``"rc"``, ``"tc"``; the default is ``"sobol"``.
              - **quantilei**: Input-magnitude scaling quantile; the default is ``1``.
              - **widthw**: Sign-magnitude width for weight values; the default is ``8``.
              - **rngw**: Weight RNG name, with the same choices as **rngi**; the default is ``"sobol"``.
              - **quantilew**: Weight-magnitude scaling quantile; the default is ``1``.
              - **cycle**: Active unary cycles, an integer of at least 1 capped at ``2 ** (widthi - 1)``, where ``None`` selects the cap. A shorter run lowers the magnitude bitwidth to ``(cycle - 1).bit_length()`` and keeps exactly ``cycle`` magnitude levels, so operands are quantized more coarsely instead of being clipped. The default is ``None``.
              - **rounding**: Dynamic-scaling rounding mode; the default is ``"round"``.
              - **name**: Optional instance label.

        The value map is persistent non-trainable state; weights and optional bias
        are trainable parameters.
        """
        super().__init__(config, [], optional_key_list=list(_DEFAULT_CONFIG))
        cfg = {**_DEFAULT_CONFIG, **config}
        #: Number of features consumed by the layer.
        self.in_features = in_features
        #: Number of features produced by the layer.
        self.out_features = out_features
        #: Quantization width used for input values.
        self.widthi = _check_hub_int('widthi', cfg['widthi'])
        #: Quantization width used for weight values.
        self.widthw = cfg['widthw']
        if self.widthi != self.widthw:
            message = f'linear_hub requires widthi == widthw (got {self.widthi}, {self.widthw}).'
            logger.error(message)
            raise AssertionError(message)
        #: Number-sequence generator used for input values.
        self.rngi = cfg['rngi'].lower()
        #: Number-sequence generator used for weight values.
        self.rngw = cfg['rngw'].lower()
        #: Input-magnitude quantile used to choose the scaling shift.
        self.quantilei = cfg['quantilei']
        #: Weight-magnitude quantile used to choose the scaling shift.
        self.quantilew = cfg['quantilew']
        #: Rounding mode used when choosing the scaling shift.
        self.rounding = cfg['rounding'].lower()
        # Sign-magnitude encoding reserves one bit, so cycle_max is 2**(width-1).
        #: Maximum cycle count supported by the configured width.
        self.cycle_max = 2 ** (self.widthi - 1)
        cycle_cfg = cfg['cycle']
        cycle_req = self.cycle_max if cycle_cfg is None else min(_check_hub_int('cycle', cycle_cfg), self.cycle_max)
        width_act, cycle_act, mapcbsg = _build_hub_map(cycle_req, self.rngi, self.rngw, self.ntype)
        #: Magnitude bitwidth carried by the active cycle count.
        self.width_act = width_act
        #: Cycle count used for each unary product.
        self.cycle_act = cycle_act
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
        rshift_i, rshift_w, rshift_o = rshift_offset(input, self.weight, self.width_act, self.width_act,
                                                     self.rounding, self.quantilei, self.quantilew)
        return _linear_hub_fn.apply(input, self.weight, self.bias,
                                    rshift_i, rshift_w, rshift_o, self.cycle_act, self.mapcbsg)

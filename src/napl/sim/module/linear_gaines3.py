import torch
import math

from napl.sim.base import napl_base
from loguru import logger


class linear_gaines3(napl_base):
    """Apply a streaming Gaines ``uMUL + gADD`` fully connected layer.

    Use this UnarySim-compatible variant when weight products should share one
    conditional-spike generator. Sharing limits accumulation accuracy through
    correlation, so prefer :class:`linear` when fidelity is the main goal.

    Per timestep, weight spikes are conditionally generated from the binary weights by a
    single shared RNG, reusing ``operation.mul_csg``. Unipolar mode ANDs them with the input
    spikes, while bipolar mode takes disjoint input-``1`` and input-``0`` paths. The
    per-output product count (parallel count, plus a bias spike on the direct path) then
    goes through a Gaines-style addition:

    - ``scaled=True``: ``output_spike = count >= scale_seq[t]`` implements a
      random-comparison scaled adder. The recovered value is
      ``(W x + b) / 2 ** round(log2(entry))``, where
      ``entry = in_features + has_bias``.

    - ``scaled=False``: unipolar mode emits ``count > 0``; bipolar mode feeds
      ``2 * count - entry`` into a ``depth``-bit saturating counter seeded at
      half range and outputs its most significant bit.

    This class matches UnarySim ``GainesLinear3``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import linear_gaines3

        layer = linear_gaines3(torch.zeros(3, 2),
                               config={"polarity": "bipolar", "timestep": 4,
                                       "generator": "sobol", "scaled": True})
        output_spike = layer(torch.ones(1, 2))

    References
    ----------
    B. R. Gaines, *Stochastic Computing Systems*.
    """


    def __init__(
            self,
            weight,
            bias=None,
            config={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
            }
        ):
        """Construct the Gaines layer from external numeric parameters.

        Args:
            weight: Numeric tensor shaped ``(out_features, in_features)``.
            bias: Optional numeric tensor shaped ``(out_features,)``. Defaults to
                ``None``.
            config: Configuration mapping with **polarity** (default
                ``"bipolar"``), **timestep** (default ``256``), **generator**
                (default ``"sobol"``), **scaled** (default ``True``),
                **scale_dim** (scaled-threshold sequence dimension, default ``6``),
                and **depth** (non-scaled counter bits, default ``8``). **name**
                is an optional instance label and defaults to ``None``.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # These imports stay local to avoid the module-operation import cycle.
        from napl.sim.module.encoder import encoder, gen_num_seq
        from napl.sim.operation.mul_csg import mul_csg

        assert weight.dim() == 2, logger.error(f'linear_gaines3 weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
        #: Trainable numeric weight matrix consumed by the shared multiplier.
        self.weight = torch.nn.Parameter(weight)
        #: Optional trainable numeric bias encoded as a spike stream.
        self.bias = torch.nn.Parameter(bias) if bias is not None else None
        #: Number of output features produced by the layer.
        self.out_features = weight.shape[0]
        #: Number of input features consumed by the layer.
        self.in_features = weight.shape[1]
        #: Whether a bias spike contributes to the parallel count.
        self.has_bias = bias is not None
        #: Parallel-count fan-in, including the bias when present.
        self.entry = self.in_features + (1 if self.has_bias else 0)

        cfg = {'polarity': self.polarity, 'timestep': config['timestep'], 'generator': config['generator']}
        # All weights share one CSG on RNG dimension 1.
        #: Shared conditional-spike multiplier for all weight products.
        self.mul = mul_csg(cfg)
        if self.has_bias:
            # Bias advances every cycle on the weight RNG dimension.
            #: Encoder that converts the optional numeric bias to spikes.
            self.b_encoder = encoder({**cfg, 'dim': 1})

        #: Whether the Gaines adder uses random-threshold scaled addition.
        self.scaled = config.get('scaled', True)
        if self.scaled:
            # The scale RNG uses a dimension distinct from the operand streams.
            width = round(math.log2(self.entry))
            #: Number of entries in the scaled-adder threshold sequence.
            self.scale_len = 2 ** width
            #: Precomputed random comparison levels for scaled addition.
            self.scale_seq: torch.Tensor
            self.register_buffer('scale_seq',
                gen_num_seq({'width': width, 'generator': cfg['generator'],
                             'dim': config.get('scale_dim', 6)}).mul(self.scale_len).floor())
        else:
            depth = config.get('depth', 8)
            #: Maximum value of the non-scaled bipolar saturating counter.
            self.max_cnt = 2 ** depth - 1
            #: Half-range decision threshold and reset value for the counter.
            self.half_cnt = 2 ** (depth - 1)
            #: Non-scaled bipolar counter, expanded to the output shape on use.
            self.cnt: torch.Tensor
            self.register_buffer('cnt', torch.full((1,), float(self.half_cnt)))


    def _reset(self):
        """Reset the local non-scaled counter.

        When **scaled** is ``False``, the counter returns to half of its configured
        range. Registered child modules are reset separately by ``reset()``.
        """
        if not self.scaled:
            self.cnt.resize_(1).fill_(self.half_cnt)


    def forward(self, input_spike):
        """Process one input-spike timestep.

        Args:
            input_spike: ``0``/``1`` tensor whose last dimension is
                ``in_features``.

        Returns:
            Output spike tensor with last dimension ``out_features``.

        The call advances the shared multiplier, optional bias encoder, local
        counter when active, and ``timestep_cur``.
        """
        # Disjoint bipolar paths let mul_csg OR equal the direct-plus-inverse count.
        prod = self.mul(input_spike.unsqueeze(-2), self.weight)
        pc = prod.sum(-1, dtype=self.ntype)
        if self.has_bias:
            pc = pc + self.b_encoder(self.bias).type(self.ntype)

        if self.scaled:
            output = torch.ge(pc, self.scale_seq[(self.timestep_cur - 1) % self.scale_len])
        else:
            if self.polarity == 'unipolar':
                output = torch.gt(pc, 0)
            else:
                delta = pc.mul(2).sub_(self.entry)
                if self.cnt.shape == delta.shape:
                    self.cnt.add_(delta).clamp_(0, self.max_cnt)
                else:
                    cnt = self.cnt.add(delta).clamp(0, self.max_cnt).detach()
                    self.cnt.resize_as_(cnt).copy_(cnt)
                output = torch.gt(self.cnt, self.half_cnt)
        return output.type(self.stype)

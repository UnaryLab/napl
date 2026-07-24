import torch, math

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger


class linear_gaines3(napl_base):
    """
    Streaming Gaines fully-connected layer: uMUL + gADD.

    Per timestep, weight spikes are conditionally generated from the binary weights by a
    single shared RNG (CSG, reusing operation.mul_csg): unipolar ANDs them with the input
    spikes, bipolar takes the disjoint input-1/input-0 dual paths (XNOR form). The
    per-output product count (parallel count, plus a bias spike on the direct path) then
    goes through a Gaines-style addition:
      - scaled (default): output spike = (count >= scale_seq[t]), a random-comparison
        scaled adder; the recovered value is (W x + b) / 2**round(log2(entry)) with
        entry = in_features + has_bias.
      - non-scaled: unipolar outputs the OR of the products (count > 0); bipolar feeds
        2*count - entry into a saturating counter of `depth` bits seeded at half range
        and outputs its MSB.

    All weight streams share ONE RNG dimension (faithful to the original), so accumulation
    accuracy is limited by inter-stream correlation; prefer linear for accuracy.
    References: B. R. Gaines, "Stochastic computing systems". UnarySim: GainesLinear3.
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
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # lazy import: operation.mul_csg imports module.encoder, so importing at module top would
        # create an import cycle with module/__init__.
        from napl.sim.module.encoder import encoder, gen_num_seq
        from napl.sim.operation.mul_csg import mul_csg

        assert weight.dim() == 2, logger.error(f'linear_gaines3 weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
        self.weight = weight
        self.out_features, self.in_features = weight.shape
        self.has_bias = bias is not None
        self.entry = self.in_features + (1 if self.has_bias else 0)

        cfg = {'polarity': self.polarity, 'timestep': config['timestep'], 'generator': config['generator']}
        # uMUL: one CSG shared by all weights on the default RNG dim (dim 1), as in the original
        self.mul = mul_csg(cfg)
        if self.has_bias:
            self.bias = bias
            # bias stream advances every cycle on the same RNG dim as the weights
            self.b_encoder = encoder({**cfg, 'dim': 1})

        self.scaled = config.get('scaled', True)
        if self.scaled:
            # gADD scale RNG: width round(log2(entry)), decorrelated from the operand dims;
            # dim 6 matches the original's (rng_idx+5) with the default rng_idx=1
            width = round(math.log2(self.entry))
            self.scale_len = 2 ** width
            self.scale_seq = torch.nn.Parameter(
                gen_num_seq({'width': width, 'generator': cfg['generator'],
                             'dim': config.get('scale_dim', 6)}).data.mul(self.scale_len).floor(),
                requires_grad=False)
        else:
            depth = config.get('depth', 8)
            self.max_cnt = 2 ** depth - 1
            self.half_cnt = 2 ** (depth - 1)
            # scalar counter that broadcasts up to the output shape on the first forward
            self.cnt = torch.nn.Parameter(torch.full((1,), float(self.half_cnt)), requires_grad=False)

    def _reset(self):
        if not self.scaled:
            self.cnt.data = torch.full((1,), float(self.half_cnt), device=self.cnt.device, dtype=self.cnt.dtype)

    def forward(self, input_spike):
        # (..., 1, in) x (out, in) -> (..., out, in) product spikes; the bipolar dual paths
        # are disjoint, so the OR mul_csg returns sums to the direct + inverse path count
        prod = self.mul(input_spike.unsqueeze(-2), self.weight)
        # sum(dtype=) casts then reduces without materializing the float (..., out, in) tensor
        pc = prod.sum(-1, dtype=self.ntype)                                   # (..., out)
        if self.has_bias:
            pc = pc + self.b_encoder(self.bias).type(self.ntype)

        if self.scaled:
            output = torch.ge(pc, self.scale_seq[(self.timestep_cur - 1) % self.scale_len])
        else:
            if self.polarity == 'unipolar':
                output = torch.gt(pc, 0)
            else:
                self.cnt.data = self.cnt.add(pc.mul(2).sub(self.entry)).clamp(0, self.max_cnt)
                output = torch.gt(self.cnt, self.half_cnt)
        return output.type(self.stype)

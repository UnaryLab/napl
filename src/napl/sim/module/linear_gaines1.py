import torch
import math

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger


class linear_gaines1(napl_base):
    """
    Streaming Gaines fully-connected layer: gMUL + gADD. Each timestep the weights (and
    bias) are encoded into spikes on a distinct RNG dimension from the input (so the
    operand streams are decorrelated), multiplied with the incoming input spikes (AND for
    unipolar, XNOR for bipolar), and the per-timestep parallel count is reduced to one
    output spike by a Gaines adder instead of linear's scaled accumulator:
    - scaled (default): the count is compared against a random level drawn from
      [0, 2**w) with w = round(log2(entry)), entry = in_features + has_bias, so the
      decoded output represents (W x + b) / 2**w for unipolar and
      (entry + W x + b) / 2**w - 1 for bipolar (= (W x + b)/entry when entry is a
      power of two).
    - non-scaled: unipolar emits count > 0 (an OR, accurate only for small inputs);
      bipolar drives a saturating up/down counter of `depth` bits by 2*count - entry,
      so the decoded output tracks clamp(W x + b, -1, 1).
    Rate-coded weights. References: Gaines 1969 stochastic computing. UnarySim: GainesLinear1.
    """
    def __init__(
            self,
            weight,
            bias=None,
            config={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
                'dim': 2,
                'scaled': True,
                'depth': 8,
            }
        ):
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # lazy import: operation.mul_csg imports module.encoder, so importing module.encoder at
        # module top would create an import cycle once this file is wired into module/__init__.
        from napl.sim.module.encoder import encoder, gen_num_seq

        assert weight.dim() == 2, logger.error(f'linear_gaines1 weight must be 2D (out_features, in_features), got {tuple(weight.shape)}.')
        self.weight = weight
        self.out_features, self.in_features = weight.shape
        self.has_bias = bias is not None
        self.entry = self.in_features + (1 if self.has_bias else 0)
        self.scaled = config.get('scaled', True)

        dim = config.get('dim', 2)
        # weight encoder on its own RNG dim; input is encoded by the caller on a different dim.
        # NB: decorrelation-by-dim only works for the sobol family; lfsr/tc/temporal ignore
        # dim and yield identical sequences across operands, biasing the result.
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'linear_gaines1 decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim (identical sequences across '
                f'operands). Use a sobol-family generator, or decorrelate the input and weight '
                f'streams by distinct seeds.')
        self.w_encoder = encoder({'polarity': self.polarity, 'timestep': config['timestep'],
                                  'generator': config['generator'], 'dim': dim})
        if self.has_bias:
            self.bias = bias
            self.b_encoder = encoder({'polarity': self.polarity, 'timestep': config['timestep'],
                                      'generator': config['generator'], 'dim': dim + 1})

        if self.scaled:
            # Gaines scaled add: compare the parallel count against a random integer level
            # in [0, 2**w); >= so a full count of 2**w always fires (matches GainesLinear1).
            self.scale_width = round(math.log2(self.entry))
            self.scale_len = 2 ** self.scale_width
            # kept as a python float list: a per-timestep scalar compare avoids indexing a
            # CPU-resident tensor (and a cross-device 0-dim broadcast) in the hot loop
            self.scale_seq = torch.floor(gen_num_seq({
                'width': self.scale_width, 'generator': config['generator'],
                'dim': dim + 2}) * self.scale_len).tolist()
        else:
            depth = config.get('depth', 8)
            self.cnt_max = 2 ** depth - 1
            self.cnt_half = 2 ** (depth - 1)
            if self.polarity == 'bipolar':
                # saturating up/down counter; scalar that broadcasts to (..., out_features)
                # on the first forward()
                self.cnt = torch.nn.Parameter(
                    torch.zeros(1, dtype=self.ntype).fill_(self.cnt_half), requires_grad=False)


    def _reset(self):
        if not self.scaled and self.polarity == 'bipolar':
            self.cnt.data = torch.zeros(
                1, dtype=self.ntype, device=self.cnt.device).fill_(self.cnt_half)


    def forward(self, input_spike):
        # input_spike: (..., in_features) spike tensor for the current timestep
        w_spike = self.w_encoder(self.weight)   # (out_features, in_features)
        xf = input_spike.type(self.ntype)
        wf = w_spike.type(self.ntype)
        # parallel count of AND (unipolar) / XNOR (bipolar) spike products, without
        # materializing the (..., out, in) elementwise product; bit-exact (small integers)
        pc = torch.matmul(xf, wf.t())           # (..., out_features)
        # pc is freshly allocated (matmul output) and stays ntype throughout, so in-place
        # updates below are alias-safe and promotion-free; they drop 4 temporaries/timestep
        if self.polarity == 'bipolar':
            pc.mul_(2).sub_(xf.sum(-1, keepdim=True)).sub_(wf.sum(-1)).add_(self.in_features)
        if self.has_bias:
            pc.add_(self.b_encoder(self.bias).type(self.ntype))   # bias spike joins the count

        if self.scaled:
            level = self.scale_seq[(self.timestep_cur - 1) % self.scale_len]
            output = torch.ge(pc, level)
        elif self.polarity == 'unipolar':
            output = torch.gt(pc, 0)
        else:
            delta = pc.mul_(2).sub_(self.entry)
            if self.cnt.shape == delta.shape:
                # steady state: in-place add/clamp (both ntype, no promotion; nothing
                # aliases cnt across timesteps)
                self.cnt.data.add_(delta).clamp_(0, self.cnt_max)
            else:
                # first timestep after reset: out-of-place op broadcasts (1,) -> (..., out)
                self.cnt.data = self.cnt.add(delta).clamp(0, self.cnt_max)
            output = torch.gt(self.cnt, self.cnt_half)
        return output.type(self.stype)

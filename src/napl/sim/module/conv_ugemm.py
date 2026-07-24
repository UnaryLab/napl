import torch, math

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger


class conv_ugemm(napl_base):
    """
    Streaming unary conv2d with uGEMM-style conditional spike generation (CSG).
    Unlike `conv` (free-running weight encoder on a distinct RNG dimension), each
    weight bitstream index advances by the incoming input spike (the `mul_csg` idiom),
    so the input/weight products are decorrelated by construction. Per timestep the
    im2col'd input spikes gate the weight-CSG bits (bipolar adds the inverse path on
    the input-0 bits), an optional bias spike joins the sum, and a scaled unary adder
    emits the output spike, folded back to NCHW. The decoded output represents
    (conv2d(x, W) + b) / scale with scale defaulting to in*kh*kw + has_bias. Bipolar
    zero-padding alternates 0/1 pads each timestep (a deterministic rate-0.5 stream =
    bipolar zero, matching the UnarySim original, unlike conv's decorrelated pad
    encoder). Rate-coded. groups=1, zero padding, scaled output only. References:
    uGEMM. UnarySim: FSUConv2duGEMM (scaled=True).
    """
    def __init__(self, weight, bias=None, stride=1, padding=0, dilation=1,
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol',
                         'scale': None, 'width': 12}):
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # lazy imports: operation.mul_csg imports module.encoder, so importing at module top
        # would create an import cycle with module/__init__.
        from napl.sim.operation import add_any
        from napl.sim.module.encoder import gen_num_seq

        assert weight.dim() == 4, logger.error(
            f'conv_ugemm weight must be 4D (out,in,kh,kw), got {tuple(weight.shape)}.')
        self.out_channels, self.in_channels = weight.shape[0], weight.shape[1]
        self.kernel_size = (weight.shape[2], weight.shape[3])
        self.stride, self.dilation = stride, dilation
        self.padding = num2tuple(padding)
        self.has_bias = bias is not None
        self.K = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.entry = self.K + (1 if self.has_bias else 0)
        scale = config.get('scale', None)
        self.scale = self.entry if scale is None else scale
        self._is_bipolar = (self.polarity == 'bipolar')

        width = config.get('width', 12)
        assert 2 ** (width - 1) >= self.entry, logger.error(
            f'conv_ugemm accumulator width <{width}> too small for fan-in <{self.entry}>: '
            f'2**(width-1) must be >= entry or partial sums saturate. Increase width.')

        self.timestep = config['timestep']
        assert self.timestep > 0, logger.error(
            f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        rng_width = math.ceil(math.log2(self.timestep))
        self.len = 2 ** rng_width
        # single RNG sequence shared by the weight/bias CSG paths (UnarySim RNG dim=1)
        self.num_seq = gen_num_seq(config={'width': rng_width, 'generator': config['generator']})

        # weight/bias as spike probabilities (bipolar maps [-1,1] -> [0,1])
        w_prob = (weight + 1) / 2 if self._is_bipolar else weight
        self.w_prob = torch.nn.Parameter(w_prob.reshape(self.out_channels, -1).type(self.ntype),
                                         requires_grad=False)                  # (out, K)
        if self.has_bias:
            b_prob = (bias + 1) / 2 if self._is_bipolar else bias
            self.b_prob = torch.nn.Parameter(b_prob.type(self.ntype), requires_grad=False)

        # CSG seq indices, advanced by the input spike (inverse path by its complement);
        # scalar start, broadcast up to (batch*L, K) on the first forward
        self.w_idx = torch.nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)
        if self._is_bipolar:
            self.w_idx_inv = torch.nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)

        self.acc = add_any({'polarity': self.polarity, 'scale': self.scale, 'width': width})
        self._im2col_key = None  # (shape, device) the cached gather indices were built for

    def _build_im2col(self, input_spike):
        # cache the unfold gather indices + output shape for this input geometry, so the
        # per-timestep loop does one index_select instead of unfold + transpose + reshape
        ph, pw = self.padding
        self._out_hw = conv2d_output_shape((input_spike.size(2), input_spike.size(3)),
                                           kernel_size=self.kernel_size, dilation=self.dilation,
                                           pad=self.padding, stride=self.stride)
        c, hp, wp = input_spike.size(1), input_spike.size(2) + 2 * ph, input_spike.size(3) + 2 * pw
        # index map via unfold on an arange (float64 on CPU: exact; MPS lacks float64)
        ar = torch.arange(c * hp * wp, dtype=torch.float64).view(1, c, hp, wp)
        u = torch.nn.functional.unfold(ar, self.kernel_size, self.dilation, 0, self.stride)  # (1, K, L)
        # L-major order so a flat gather yields the (P, K) layout directly
        self._im2col_idx = u.view(self.K, -1).t().contiguous().long().view(-1).to(input_spike.device)
        self._im2col_key = (input_spike.shape, input_spike.device)

    def _reset(self):
        self.w_idx.data = torch.zeros(1, dtype=torch.long, device=self.w_idx.device)
        if self._is_bipolar:
            self.w_idx_inv.data = torch.zeros(1, dtype=torch.long, device=self.w_idx_inv.device)

    def forward(self, input_spike):
        # input_spike: (batch, in_channels, H, W) spike tensor for the current timestep
        ph, pw = self.padding
        if self._im2col_key != (input_spike.shape, input_spike.device):
            self._build_im2col(input_spike)
        # gather indices are dtype-agnostic; cast to float for the RNG-compare arithmetic
        xf = input_spike.type(self.ntype)
        if self.padding != (0, 0):
            # bipolar zero = rate-0.5: alternate 0/1 pads, 0 first (FSUConv2duGEMM's
            # even_cycle_flag); unipolar always pads 0
            pad_bit = float((self.timestep_cur - 1) % 2) if self._is_bipolar else 0.0
            xf = torch.nn.functional.pad(xf, (pw, pw, ph, ph), value=pad_bit)
        # cached-index gather == unfold + transpose + reshape (one copy instead of three)
        inp = xf.reshape(xf.size(0), -1).index_select(1, self._im2col_idx).view(-1, self.K)  # (P, K)
        inv = 1 - inp

        # input-1 path: gate the CSG compare by the input bit -- input-0 lanes see rnd+2,
        # above any probability, so they never fire -- then count over the fan-in
        # (bool sum with dtype= fuses the cast, skipping the (P, out, K) float intermediate)
        rnd = self.num_seq[self.w_idx] + inv * 2                              # (P, K)
        psum = torch.gt(self.w_prob.unsqueeze(0), rnd.unsqueeze(1)).sum(-1, dtype=self.ntype)  # (P, out)
        # conditional seq-index update: advances only where the input spike is 1
        # (in-place once broadcast to (P, K); first call must expand out-of-place)
        if self.w_idx.shape == inp.shape:
            self.w_idx.add_(inp.type(torch.long))
        else:
            self.w_idx.data = self.w_idx.add(inp.type(torch.long))

        if self._is_bipolar:
            # input-0 path: (1-x) & (1 - w_bit); input-1 lanes see rnd-2, below any
            # probability, so they never fire; index advances only where the input is 0
            rnd_inv = self.num_seq[self.w_idx_inv] - inp * 2                  # (P, K)
            psum += torch.le(self.w_prob.unsqueeze(0), rnd_inv.unsqueeze(1)).sum(-1, dtype=self.ntype)
            if self.w_idx_inv.shape == inv.shape:
                self.w_idx_inv.add_(inv.type(torch.long))
            else:
                self.w_idx_inv.data = self.w_idx_inv.add(inv.type(torch.long))

        if self.has_bias:
            # bias stream is free-running on the shared RNG (index = timestep)
            psum += torch.gt(self.b_prob,
                             self.num_seq[(self.timestep_cur - 1) % self.len]).type(self.ntype)

        acc = self.acc(psum, entry=self.entry, dim=None)                      # (P, out) output spikes
        # fold with a (1,1) kernel is a pure reshape; acc is already stype, no casts needed
        return acc.view(input_spike.size(0), -1, acc.size(-1)).transpose(1, 2) \
                  .reshape(input_spike.size(0), acc.size(-1), *self._out_hw)  # (batch, out, H, W)

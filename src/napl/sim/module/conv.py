import torch
import math

from napl.utils import *
from napl.sim.base import napl_base
from napl.sim.module.linear import (_linear_fxp_fn, _linear_hub_fn,
                                _linear_tlut_fxpfxp_fn, _linear_tlut_fxpfp_fn, _linear_tlut_fpfp_fn,
                                _build_hub_map, _TLUT_FP_WIDTH)
from loguru import logger


def _init_conv_params(module, in_channels, out_channels, kernel_size, bias, weight_ext, bias_ext):
    """Give `module` nn.Conv2d-style learnable weight/bias (or adopt external tensors)."""
    kh, kw = num2tuple(kernel_size)
    module.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
    torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    if bias:
        module.bias = torch.nn.Parameter(torch.empty(out_channels))
        bound = 1 / math.sqrt(in_channels * kh * kw)
        torch.nn.init.uniform_(module.bias, -bound, bound)
    else:
        module.bias = None
    if weight_ext is not None:
        assert tuple(weight_ext.shape) == (out_channels, in_channels, kh, kw), \
            logger.error(f'weight_ext shape {tuple(weight_ext.shape)} != {(out_channels, in_channels, kh, kw)}.')
        module.weight.data = weight_ext.clone().type(module.weight.dtype)
    if bias and bias_ext is not None:
        assert tuple(bias_ext.shape) == (out_channels,), \
            logger.error(f'bias_ext shape {tuple(bias_ext.shape)} != ({out_channels},).')
        module.bias.data = bias_ext.clone().type(module.bias.dtype)


def _conv2d_binary(input, weight, bias, kernel_size, stride, padding, dilation, linear_fn):
    """
    Run a binary-domain conv2d by im2col + a 2D linear kernel + fold: unfold the input to
    patches, apply `linear_fn(patches_2d, weight_2d)` (one of the M2 binary linear autograd
    Functions), then fold the result back to NCHW. Bias is added after folding. The unfold/
    fold are differentiable, so the linear Function's STE gradient flows through to weight
    and input.
    """
    out_hw = conv2d_output_shape((input.size(2), input.size(3)), kernel_size=kernel_size,
                                 dilation=dilation, pad=padding, stride=stride)
    im2col = torch.nn.functional.unfold(input, kernel_size, dilation, padding, stride)   # (b, K, L)
    inp2d = im2col.transpose(1, 2).reshape(-1, im2col.size(1))                            # (b*L, K)
    w2d = weight.view(weight.size(0), -1)                                                 # (out, K)
    mm = linear_fn(inp2d, w2d)                                                            # (b*L, out)
    mm = mm.reshape(input.size(0), -1, mm.size(-1)).transpose(1, 2)                       # (b, out, L)
    # fold with a (1,1) kernel is exactly a row-major reshape of L -> (H, W); reshape is
    # cheaper and has the identical (unfold==reshape) gradient
    out = mm.reshape(input.size(0), mm.size(1), out_hw[0], out_hw[1])                     # (b, out, H, W)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


class conv_fxp(napl_base):
    """
    Binary-domain fixed-point conv2d (im2col + linear_fxp kernel + fold). Single-shot, no
    tick; trains via STE. Approximates nn.Conv2d within the quantization bound.
    groups=1, zero padding only.
    """
    streaming = False
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bias=True, weight_ext=None, bias_ext=None,
                 config={'widthi': 8, 'quantilei': 1, 'widthw': 8, 'quantilew': 1, 'rounding': 'round'}):
        super().__init__(config, [])
        self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation
        self.widthi = config.get('widthi', 8)
        self.widthw = config.get('widthw', 8)
        self.quantilei = config.get('quantilei', 1)
        self.quantilew = config.get('quantilew', 1)
        self.rounding = config.get('rounding', 'round').lower()
        self.max_abs_i = 2 ** self.widthi
        self.max_abs_w = 2 ** self.widthw
        _init_conv_params(self, in_channels, out_channels, kernel_size, bias, weight_ext, bias_ext)

    def forward(self, input):
        rshift_i, rshift_w, _ = rshift_offset(input, self.weight, self.widthi - 1, self.widthw - 1,
                                              self.rounding, self.quantilei, self.quantilew)
        rshift_o = 0 - rshift_i - rshift_w
        fn = lambda i2d, w2d: _linear_fxp_fn.apply(i2d, w2d, None, rshift_i, rshift_w, rshift_o,
                                                   self.max_abs_i, self.max_abs_w)
        return _conv2d_binary(input, self.weight, self.bias, self.kernel_size, self.stride,
                              self.padding, self.dilation, fn)


class conv_hub(napl_base):
    """
    Binary-domain HUB conv2d (im2col + linear_hub value-map kernel + fold). Single-shot, no
    tick; trains via STE. Approximates nn.Conv2d within the unary-multiplication bound.
    groups=1, zero padding only, requires widthi==widthw.
    """
    streaming = False
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bias=True, weight_ext=None, bias_ext=None,
                 config={'widthi': 8, 'rngi': 'sobol', 'quantilei': 1, 'widthw': 8, 'rngw': 'sobol',
                         'quantilew': 1, 'cycle': 128, 'rounding': 'round'}):
        super().__init__(config, [])
        self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation
        self.widthi = config.get('widthi', 8)
        self.widthw = config.get('widthw', 8)
        assert self.widthi == self.widthw, \
            logger.error(f'conv_hub requires widthi == widthw (got {self.widthi}, {self.widthw}).')
        self.rngi = config.get('rngi', 'sobol').lower()
        self.rngw = config.get('rngw', 'sobol').lower()
        self.quantilei = config.get('quantilei', 1)
        self.quantilew = config.get('quantilew', 1)
        self.rounding = config.get('rounding', 'round').lower()
        self.cycle_max, mapcbsg = _build_hub_map(self.widthi, self.widthw, self.rngi, self.rngw, self.ntype)
        cycle_cfg = config.get('cycle', None)
        self.cycle_act = self.cycle_max if cycle_cfg is None else min(cycle_cfg, self.cycle_max)
        self.mapcbsg = torch.nn.Parameter(mapcbsg, requires_grad=False)
        _init_conv_params(self, in_channels, out_channels, kernel_size, bias, weight_ext, bias_ext)

    def forward(self, input):
        rshift_i, rshift_w, rshift_o = rshift_offset(input, self.weight, self.widthi - 1, self.widthw - 1,
                                                     self.rounding, self.quantilei, self.quantilew)
        fn = lambda i2d, w2d: _linear_hub_fn.apply(i2d, w2d, None, rshift_i, rshift_w, rshift_o,
                                                   self.cycle_act, self.mapcbsg)
        return _conv2d_binary(input, self.weight, self.bias, self.kernel_size, self.stride,
                              self.padding, self.dilation, fn)


class conv_tlut(napl_base):
    """
    Binary-domain temporal-LUT conv2d (im2col + linear_tlut kernel + fold), modes
    fxpfxp/fxpfp/fpfp via the (formati, formatw) pair. Single-shot; trains via STE.
    Approximates nn.Conv2d within the temporal-decomposition bound.
    groups=1, zero padding only.
    """
    streaming = False
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bias=True, weight_ext=None, bias_ext=None,
                 config={'temporal': 'i', 'widtht': 4, 'formati': 'fxp', 'widthi': 8, 'quantilei': 1,
                         'formatw': 'fxp', 'widthw': 8, 'quantilew': 1, 'cycle': None, 'rounding': 'round'}):
        super().__init__(config, [])
        self.kernel_size, self.stride, self.padding, self.dilation = kernel_size, stride, padding, dilation
        self.temporal = config.get('temporal', 'i').lower()
        self.widtht = config.get('widtht', 4)
        self.formati = config.get('formati', 'fxp').lower()
        self.formatw = config.get('formatw', 'fxp').lower()
        self.widthi = config.get('widthi', 8)
        self.widthw = config.get('widthw', 8)
        self.quantilei = config.get('quantilei', 1)
        self.quantilew = config.get('quantilew', 1)
        self.rounding = config.get('rounding', 'round').lower()
        assert self.temporal in ('i', 'input', 'w', 'weight'), \
            logger.error(f"conv_tlut 'temporal' must be one of ['i','input','w','weight'], got {self.temporal}.")

        if self.formati == 'fxp' and self.formatw == 'fxp':
            self.mode = 'fxpfxp'
        elif self.formati != 'fxp' and self.formatw != 'fxp':
            self.mode = 'fpfp'
        else:
            self.mode = 'fxpfp'

        self.cycle_max = 2 ** self.widtht
        cycle_cfg = config.get('cycle', None)
        self.cycle_act = self.cycle_max if cycle_cfg is None else min(cycle_cfg, self.cycle_max)
        self.widthi_mag = self.widthi - 1
        self.widthw_mag = self.widthw - 1
        if self.temporal in ('i', 'input'):
            fmt = self.formati
            self.width = self.widthi - 1 if fmt == 'fxp' else _TLUT_FP_WIDTH[fmt]
        else:
            fmt = self.formatw
            self.width = self.widthw - 1 if fmt == 'fxp' else _TLUT_FP_WIDTH[fmt]
        self.degree = int(math.ceil(self.width / self.widtht))
        self.delta = int(self.degree * self.widtht - self.width)
        _init_conv_params(self, in_channels, out_channels, kernel_size, bias, weight_ext, bias_ext)

    def forward(self, input):
        cp, cn = self.cycle_act, -self.cycle_act
        if self.mode == 'fxpfxp':
            fn = lambda i2d, w2d: _linear_tlut_fxpfxp_fn.apply(i2d, w2d, None, self.temporal, self.widthi_mag,
                self.widthw_mag, self.widtht, self.degree, self.delta, cp, cn, self.rounding, self.quantilei, self.quantilew)
        elif self.mode == 'fxpfp':
            fn = lambda i2d, w2d: _linear_tlut_fxpfp_fn.apply(i2d, w2d, None, self.temporal, self.width,
                self.widtht, self.degree, self.delta, cp, cn, self.rounding, self.quantilei, self.quantilew)
        else:
            fn = lambda i2d, w2d: _linear_tlut_fpfp_fn.apply(i2d, w2d, None, self.temporal, self.width,
                self.widtht, self.degree, self.delta, cp, cn)
        return _conv2d_binary(input, self.weight, self.bias, self.kernel_size, self.stride,
                              self.padding, self.dilation, fn)


class conv(napl_base):
    """
    Streaming unary conv2d, computed bit by bit. Each timestep the weights are encoded
    into spikes on a distinct RNG dimension, multiplied (XNOR bipolar / AND unipolar) with the
    im2col'd input patches, and the partial products summed by a scaled unary adder, then
    folded back to NCHW. The decoded output represents conv2d(x, W) + b divided by `scale`
    (default in_channels*kh*kw + has_bias) to stay in unary range. Bipolar zero-padding uses a
    decorrelated rate-0.5 pad stream (a separate pad encoder), not a deterministic toggle.
    Rate-coded weights. groups=1, zero padding only.
    """
    def __init__(self, weight, bias=None, stride=1, padding=0, dilation=1,
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol',
                         'dim': 2, 'scale': None, 'width': 12}):
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        from napl.sim.operation import add_any
        from napl.sim.module.encoder import encoder

        assert weight.dim() == 4, logger.error(f'conv weight must be 4D (out,in,kh,kw), got {tuple(weight.shape)}.')
        self.weight = weight
        self.out_channels, self.in_channels = weight.shape[0], weight.shape[1]
        self.kernel_size = (weight.shape[2], weight.shape[3])
        self.stride, self.dilation = stride, dilation
        self.padding = num2tuple(padding)
        self.has_bias = bias is not None
        self.weight_flat = weight.view(self.out_channels, -1)                 # (out, K)
        self.K = self.weight_flat.shape[1]                                    # in*kh*kw
        self.entry = self.K + (1 if self.has_bias else 0)
        scale = config.get('scale', None)
        self.scale = self.entry if scale is None else scale

        width = config.get('width', 12)
        assert 2 ** (width - 1) >= self.entry, logger.error(
            f'conv accumulator width <{width}> too small for fan-in <{self.entry}>: '
            f'2**(width-1) must be >= entry or partial sums saturate. Increase width.')
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'conv decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim. Use a sobol-family generator.')

        dim = config.get('dim', 2)
        cfg = {'polarity': self.polarity, 'timestep': config['timestep'], 'generator': config['generator']}
        self.w_encoder = encoder({**cfg, 'dim': dim})
        self.acc = add_any({'polarity': self.polarity, 'scale': self.scale, 'width': width})
        if self.has_bias:
            self.bias = bias
            self.b_encoder = encoder({**cfg, 'dim': dim + 1})
        # decorrelated rate-0.5 pad stream for bipolar zero-padding
        if self.polarity == 'bipolar' and self.padding != (0, 0):
            self.pad_encoder = encoder({**cfg, 'dim': dim + 2})
            # The pad bit feeds F.pad's scalar `value`, so it must be a Python float; taking
            # .item() of a fresh encoder spike every timestep forces a per-step device->host
            # sync (costly on GPU). The bit (rate-0.5 spike of input 0) is a deterministic
            # function of timestep, so precompute the full period to plain floats once.
            self.pad_len = self.pad_encoder.len
            pad_seq = torch.gt(torch.tensor(0.5, dtype=self.ntype),
                               self.pad_encoder.num_seq.detach()).type(self.stype)
            self.pad_bits = [float(b) for b in pad_seq.tolist()]

    def forward(self, input_spike):
        # input_spike: (batch, in_channels, H, W) spike tensor for the current timestep
        ph, pw = self.padding
        out_hw = conv2d_output_shape((input_spike.size(2), input_spike.size(3)), kernel_size=self.kernel_size,
                                     dilation=self.dilation, pad=self.padding, stride=self.stride)
        # unfold/fold are not implemented for int8 spikes; cast to float (spikes are 0/1, exact)
        xf = input_spike.type(self.ntype)
        if self.polarity == 'bipolar' and self.padding != (0, 0):
            # pad with a decorrelated rate-0.5 spike (bipolar 0), then unfold with no extra padding
            pad_bit = self.pad_bits[(self.timestep_cur - 1) % self.pad_len]
            xf = torch.nn.functional.pad(xf, (pw, pw, ph, ph), value=pad_bit)
            im2col = torch.nn.functional.unfold(xf, self.kernel_size, self.dilation, 0, self.stride)
        else:
            im2col = torch.nn.functional.unfold(xf, self.kernel_size, self.dilation, self.padding, self.stride)
        w_spike = self.w_encoder(self.weight_flat)                           # (out, K)
        wf = w_spike.type(self.ntype)
        # partial sum of the AND (unipolar) / XNOR (bipolar) spike products, without
        # materializing the (batch, out, K, L) elementwise product; bit-exact (small
        # integers, exact in float under any reduction order). Batched (out,K)@(batch,K,L)
        # avoids transposing/copying the big im2col tensor.
        psum = torch.matmul(wf, im2col)                                      # (batch, out, L)
        if self.polarity == 'bipolar':
            psum = 2 * psum - im2col.sum(1, keepdim=True) - wf.sum(-1).unsqueeze(-1) + self.K
        if self.has_bias:
            psum = psum + self.b_encoder(self.bias).type(self.ntype).unsqueeze(-1)  # bias spike joins the sum
        acc = self.acc(psum, entry=self.entry, dim=None)                     # (batch, out, L) spikes
        # fold with a (1,1) kernel is exactly a row-major reshape of L -> (H, W)
        return acc.reshape(input_spike.size(0), acc.size(1), out_hw[0], out_hw[1])  # (batch, out, H, W)

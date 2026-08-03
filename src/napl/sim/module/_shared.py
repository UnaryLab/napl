import math
import torch

from napl.utils import *
from loguru import logger


def _init_linear_params(module, in_features, out_features, bias, weight_ext, bias_ext):
    """Give ``module`` ``nn.Linear``-style trainable parameters."""
    module.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
    torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    if bias:
        module.bias = torch.nn.Parameter(torch.empty(out_features))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(module.bias, -bound, bound)
    else:
        module.bias = None
    if weight_ext is not None:
        assert tuple(weight_ext.shape) == (out_features, in_features), \
            logger.error(f'weight_ext shape {tuple(weight_ext.shape)} != ({out_features}, {in_features}).')
        module.weight.data = weight_ext.clone().type(module.weight.dtype)
    if bias and bias_ext is not None:
        assert tuple(bias_ext.shape) == (out_features,), \
            logger.error(f'bias_ext shape {tuple(bias_ext.shape)} != ({out_features},).')
        module.bias.data = bias_ext.clone().type(module.bias.dtype)

def _linear_ste_grads(ctx, grad_output):
    """
    Straight-through gradient shared by the binary-domain linear kernels: the forward is
    an approximate/quantized matmul, but the backward is the exact linear gradient so the
    layer trains like a normal nn.Linear.
    """
    input, weight, bias = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None
    if ctx.needs_input_grad[0]:
        grad_input = grad_output.matmul(weight)
    if ctx.needs_input_grad[1]:
        grad_weight = grad_output.t().matmul(input)
    if bias is not None and ctx.needs_input_grad[2]:
        grad_bias = grad_output.sum(0)
    return grad_input, grad_weight, grad_bias

def _hub_rng_seq(width, rng='sobol'):
    """
    Integer RNG sequence of length 2**width in [0, 2**width).
    Used to build the HUB unary-multiplication value map.
    """
    seq_len = 2 ** width
    rng = rng.lower()
    if rng in ('sobol', 'rc'):
        seq = torch.quasirandom.SobolEngine(1).draw(seq_len)[:, 0].view(seq_len) * seq_len
    elif rng in ('race', 'tc'):
        seq = torch.tensor([x / seq_len for x in range(seq_len)]) * seq_len
    elif rng in ('race10', 'tc10'):
        seq = torch.flip(torch.tensor([x / seq_len for x in range(seq_len)]) * seq_len, [0])
    else:
        seq = torch.quasirandom.SobolEngine(1).draw(seq_len)[:, 0].view(seq_len) * seq_len
    return seq.floor()

def _build_hub_map(widthi, widthw, rngi, rngw, ntype):
    """
    Build the HUB unary-multiplication value map: mapcbsg[i_level, w_level] is the
    bitstream AND-count for an input of magnitude i_level and a weight of magnitude
    w_level, under the chosen RNGs (sign-magnitude, so cycle_max = 2**(width-1)).
    Returns (cycle_max, mapcbsg). Shared by linear_hub and conv_hub. Requires widthi==widthw.
    """
    cmax = 2 ** (max(widthi, widthw) - 1)
    rngctler = _hub_rng_seq(widthi - 1, rngi)
    rngctlee = _hub_rng_seq(widthw - 1, rngw)
    levels = torch.arange(cmax, dtype=torch.float).unsqueeze(1)
    ctler_bit = torch.gt(levels.expand(cmax, cmax), rngctler.unsqueeze(0))
    mapctler = torch.sum(ctler_bit, 1).type(torch.long)
    ctlee_bit = torch.gt(levels.expand(cmax, cmax), rngctlee.unsqueeze(0))
    mapcbsg = torch.empty(cmax, cmax, dtype=torch.long)
    for c in range(cmax):
        mapcbsg[c] = torch.sum(ctlee_bit[:, 0:mapctler[c]], 1)
    return cmax, mapcbsg.type(ntype)

def _tlut_decompose(mag, widtht, degree, cycle_neg, cycle_pos):
    """
    Temporal LUT decomposition: split a truncated fixed-point magnitude tensor into a
    sum of ``degree`` ``widtht``-bit temporal digits, each clamped to the run cycle range.
    Returns the recomposed magnitude. Shared by the TLUT forward modes.
    """
    out = torch.zeros_like(mag)
    for _ in range(degree):
        mag = pow2_rshift(mag, widtht)
        frac = torch.frac(mag)
        mag = torch.trunc(mag)
        frac = pow2_lshift(frac, widtht).clamp(cycle_neg + 1, cycle_pos - 1)
        out = pow2_rshift(frac, widtht) + pow2_rshift(out, widtht)
    return out

class _linear_fxp_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, rshift_i, rshift_w, rshift_o, max_abs_i, max_abs_w,
                full_signed_range=False):
        ctx.save_for_backward(input, weight, bias)
        bot_i = -max_abs_i if full_signed_range else 1 - max_abs_i
        top_i = max_abs_i - 1
        i_round = pow2_rshift(input, rshift_i)
        i_round.round_().clamp_(bot_i, top_i)
        bot_w = -max_abs_w if full_signed_range else 1 - max_abs_w
        top_w = max_abs_w - 1
        w_round = pow2_rshift(weight, rshift_w)
        w_round.round_().clamp_(bot_w, top_w)
        output = torch.matmul(i_round, w_round.t())
        output = pow2_rshift(output, rshift_o)
        if bias is not None:
            output = output + bias
        return output


    @staticmethod
    def backward(ctx, grad_output):
        return _linear_ste_grads(ctx, grad_output) + (None,) * (len(ctx.needs_input_grad) - 3)

class _linear_hub_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, rshift_i, rshift_w, rshift_o, cycle, mapcbsg):
        ctx.save_for_backward(input, weight, bias)
        assert input.dim() == 2, logger.error('linear_hub input needs 2 dims (batch, in_features).')
        buf_i = pow2_rshift(input, rshift_i).unsqueeze(1).abs().type(torch.long).clamp(0, cycle - 1)
        buf_w = pow2_rshift(weight, rshift_w).unsqueeze(0).abs().type(torch.long).clamp(0, cycle - 1)
        act_input = torch.sign(input).unsqueeze(1)
        act_wght = torch.sign(weight).unsqueeze(0)
        prod = mapcbsg[buf_i, buf_w].type(act_wght.dtype) * act_wght
        output = torch.matmul(act_input, prod.transpose(1, 2))
        output = pow2_rshift(output, rshift_o).squeeze(1)
        if bias is not None:
            output = output + bias.unsqueeze(0).expand_as(output)
        return output


    @staticmethod
    def backward(ctx, grad_output):
        return _linear_ste_grads(ctx, grad_output) + (None, None, None, None, None)

class _linear_tlut_fxpfxp_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, temporal, widthi, widthw, widtht, degree, delta,
                cycle_pos, cycle_neg, rounding, quantilei, quantilew):
        ctx.save_for_backward(input, weight, bias)
        in_fp = input.detach().clone().to(torch.float)
        w_fp = weight.detach().clone().to(torch.float)
        rshift_i, rshift_w, _ = rshift_offset(in_fp, w_fp, widthi, widthw, rounding, quantilei, quantilew)
        in_fp = torch.trunc(pow2_rshift(in_fp, rshift_i).clamp(-2 ** widthi + 1, 2 ** widthi - 1))
        w_fp = torch.trunc(pow2_rshift(w_fp, rshift_w).clamp(-2 ** widthw + 1, 2 ** widthw - 1))
        if temporal in ('i', 'input'):
            in_new = _tlut_decompose(in_fp, widtht, degree, cycle_neg, cycle_pos)
            input_new = pow2_lshift(in_new, delta + widthi + rshift_i).type(weight.dtype)
            weight_new = pow2_lshift(w_fp, rshift_w).type(weight.dtype)
        else:
            w_new = _tlut_decompose(w_fp, widtht, degree, cycle_neg, cycle_pos)
            input_new = pow2_lshift(in_fp, rshift_i).type(input.dtype)
            weight_new = pow2_lshift(w_new, delta + widthw + rshift_w).type(input.dtype)
        output = torch.matmul(input_new, weight_new.t())
        if bias is not None:
            output = output + bias.unsqueeze(0).expand_as(output)
        return output


    @staticmethod
    def backward(ctx, grad_output):
        return _linear_ste_grads(ctx, grad_output) + (None,) * 11

class _linear_tlut_fxpfp_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, temporal, width, widtht, degree, delta,
                cycle_pos, cycle_neg, rounding, quantilei, quantilew):
        ctx.save_for_backward(input, weight, bias)
        in_fp = input.detach().clone().to(torch.float)
        w_fp = weight.detach().clone().to(torch.float)
        rshift_i, rshift_w, _ = rshift_offset(in_fp, w_fp, width, width, rounding, quantilei, quantilew)
        if temporal in ('i', 'input'):
            in_fp = torch.trunc(pow2_rshift(in_fp, rshift_i).clamp(-2 ** width + 1, 2 ** width - 1))
            in_new = _tlut_decompose(in_fp, widtht, degree, cycle_neg, cycle_pos)
            input_new = pow2_lshift(in_new, delta + width + rshift_i).type(weight.dtype)
            weight_new = weight
        else:
            w_fp = torch.trunc(pow2_rshift(w_fp, rshift_w).clamp(-2 ** width + 1, 2 ** width - 1))
            w_new = _tlut_decompose(w_fp, widtht, degree, cycle_neg, cycle_pos)
            input_new = input
            weight_new = pow2_lshift(w_new, delta + width + rshift_w).type(input.dtype)
        output = torch.matmul(input_new, weight_new.t())
        if bias is not None:
            output = output + bias.unsqueeze(0).expand_as(output)
        return output


    @staticmethod
    def backward(ctx, grad_output):
        return _linear_ste_grads(ctx, grad_output) + (None,) * 10

class _linear_tlut_fpfp_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, temporal, width, widtht, degree, delta, cycle_pos, cycle_neg):
        ctx.save_for_backward(input, weight, bias)
        dtype = input.dtype
        src = (input if temporal in ('i', 'input') else weight).detach().clone().to(torch.float)
        try:
            mantissa, exponent = torch.frexp(src)
        except NotImplementedError:
            # MPS lacks frexp; CPU results are bit-exact after transfer back.
            mantissa, exponent = torch.frexp(src.cpu())
            mantissa, exponent = mantissa.to(src.device), exponent.to(src.device)
        mantissa = _tlut_decompose(pow2_lshift(mantissa, width), widtht, degree, cycle_neg, cycle_pos)
        mantissa = pow2_lshift(mantissa, delta)
        recomposed = torch.ldexp(mantissa, exponent).type(dtype)
        if temporal in ('i', 'input'):
            input_new, weight_new = recomposed, weight
        else:
            input_new, weight_new = input, recomposed
        output = torch.matmul(input_new, weight_new.t())
        if bias is not None:
            output = output + bias.unsqueeze(0).expand_as(output)
        return output


    @staticmethod
    def backward(ctx, grad_output):
        return _linear_ste_grads(ctx, grad_output) + (None,) * 7

_TLUT_FP_WIDTH = {'bfloat16': 8, 'float16': 11, 'float32': 24}

def _init_conv_params(module, in_channels, out_channels, kernel_size, bias, weight_ext, bias_ext):
    """Give ``module`` ``nn.Conv2d``-style trainable parameters."""
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
    patches, apply ``linear_fn(patches_2d, weight_2d)`` through a binary linear autograd
    Functions), then fold the result back to NCHW. Bias is added after folding. The unfold/
    fold are differentiable, so the linear Function's STE gradient flows through to weight
    and input.
    """
    out_hw = conv2d_output_shape((input.size(2), input.size(3)), kernel_size=kernel_size,
                                 dilation=dilation, pad=padding, stride=stride)
    im2col = torch.nn.functional.unfold(input, kernel_size, dilation, padding, stride)
    inp2d = im2col.transpose(1, 2).reshape(-1, im2col.size(1))
    w2d = weight.view(weight.size(0), -1)
    mm = linear_fn(inp2d, w2d)
    mm = mm.reshape(input.size(0), -1, mm.size(-1)).transpose(1, 2)
    out = mm.reshape(input.size(0), mm.size(1), out_hw[0], out_hw[1])
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out

def _init_mgu_params(module, input_size, hidden_size, bias):
    """MGU forget/new-gate weights/biases, truncated-normal init."""
    module.weight_f = torch.nn.Parameter(torch.empty(hidden_size, hidden_size + input_size))
    module.weight_n = torch.nn.Parameter(torch.empty(hidden_size, hidden_size + input_size))
    if bias:
        module.bias_f = torch.nn.Parameter(torch.empty(hidden_size))
        module.bias_n = torch.nn.Parameter(torch.empty(hidden_size))
    else:
        module.register_parameter('bias_f', None)
        module.register_parameter('bias_n', None)
    stdv = 1.0 / math.sqrt(hidden_size)
    for w in [module.weight_f, module.weight_n, module.bias_f, module.bias_n]:
        if w is not None:
            w.data = truncated_normal(w, 0.0, stdv)

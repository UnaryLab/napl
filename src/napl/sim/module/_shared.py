import math
import torch

from napl.utils import (
    num2tuple,
    pow2_rshift,
    truncated_normal,
)
from loguru import logger


def _check_acc_width(name, width, entry, scale, polarity):
    """
    Check the signed accumulator width of the streaming unary adder in module ``name``
    and return it. Before thresholding, the accumulator holds the largest sub-threshold
    residue (scale - grid) plus one timestep's step, which the bipolar offset
    (entry - scale)/2 halves. For scale < entry the accumulator drains by at most scale
    per timestep, so the width also carries a burst-headroom floor that absorbs one
    worst-case timestep.
    """
    if not isinstance(width, int):
        message = f'{name} accumulator width must be int: got <{width}>.'
        logger.error(message)
        raise AssertionError(message)
    delta_max = (entry + scale) / 2 if polarity == 'bipolar' else entry
    grid = 0.5 if polarity == 'bipolar' and (entry - scale) % 2 else 1
    if (2 ** (width - 1) - 1 < (scale - grid) + delta_max
            or (scale < entry and 2 ** (width - 1) <= entry)):
        message = (
            f'{name} accumulator width <{width}> too small for fan-in <{entry}> '
            f'and scale <{scale}>: 2**(width-1) - 1 must be >= (scale - grid) + '
            f'delta_max, with grid <{grid}> and delta_max <{delta_max}> for '
            f'<{polarity}>, and 2**(width-1) must be > entry when scale < entry, '
            f'or partial sums saturate. Increase width.'
        )
        logger.error(message)
        raise AssertionError(message)
    return width


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
        if tuple(weight_ext.shape) != (out_features, in_features):
            message = f'weight_ext shape {tuple(weight_ext.shape)} != ({out_features}, {in_features}).'
            logger.error(message)
            raise AssertionError(message)
        module.weight.data = weight_ext.clone().type(module.weight.dtype)
    if bias and bias_ext is not None:
        if tuple(bias_ext.shape) != (out_features,):
            message = f'bias_ext shape {tuple(bias_ext.shape)} != ({out_features},).'
            logger.error(message)
            raise AssertionError(message)
        module.bias.data = bias_ext.clone().type(module.bias.dtype)


def _gaines_counter_step(cnt, delta, cnt_max, cnt_half):
    """
    Advance the non-scaled Gaines bipolar saturating counter by one timestep and return
    its output spike. With ``a`` the counter and ``d`` the per-timestep increment,

    .. math::

       a_t = \\mathrm{clamp}(a_{t-1} + d_t,\\; 0,\\; \\mathit{cnt\\_max}),\\qquad
       y_t = \\mathbf{1}\\{a_t > \\mathit{cnt\\_half}\\}.

    The scalar initial state broadcasts out of place on the first call; matching shapes
    update in place. Shared by linear_gaines1 and linear_gaines2.
    """
    if cnt.shape == delta.shape:
        cnt.add_(delta).clamp_(0, cnt_max)
    else:
        expanded = cnt.add(delta).clamp(0, cnt_max).detach()
        cnt.resize_as_(expanded).copy_(expanded)
    return torch.gt(cnt, cnt_half)


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


def rshift_offset(input, weight, widthi, widthw, rounding="round", quantilei=1, quantilew=1):
    """
    Dynamic fixed-point scaling: return the right-shift offsets that bring `input` and
    `weight` into a `widthi`/`widthw`-bit range (from their quantile-clipped magnitude),
    plus the output offset that undoes both.
    """
    def _mag(x, q):
        # q=1 bypasses torch.quantile, which rejects tensors larger than 2**24 elements.
        if q == 1:
            return x.abs().max()
        lower = torch.quantile(x, 0.5 + q / 2)
        upper = torch.quantile(x, 0.5 - q / 2)
        return torch.max(lower.abs(), upper.abs())

    with torch.no_grad():
        imax_int = _mag(input, quantilei).log2()
        wmax_int = _mag(weight, quantilew).log2()

        if rounding == "round":
            imax_int = imax_int.round()
            wmax_int = wmax_int.round()
        elif rounding == "floor":
            imax_int = imax_int.floor()
            wmax_int = wmax_int.floor()
        elif rounding == "ceil":
            imax_int = imax_int.ceil()
            wmax_int = wmax_int.ceil()

        # Zero-magnitude operands map log2(0) to a zero offset, keeping results finite.
        imax_int = torch.nan_to_num(imax_int, nan=0.0, neginf=0.0, posinf=0.0)
        wmax_int = torch.nan_to_num(wmax_int, nan=0.0, neginf=0.0, posinf=0.0)

        rshift_i = imax_int - widthi
        rshift_w = wmax_int - widthw
        rshift_o = max(widthi, widthw) - imax_int - wmax_int
        return rshift_i, rshift_w, rshift_o


def _shift_round_clamp(x, rshift, lo, hi):
    """
    Quantize ``x`` to the grid ``rshift`` places up: right-shift by ``rshift``, round to
    the nearest integer, then clamp to ``[lo, hi]``. ``rshift`` may be negative, which
    shifts left. ``pow2_rshift`` returns a fresh tensor, so the in-place rounding and
    clamping cannot modify ``x``.
    """
    out = pow2_rshift(x, rshift)
    out.round_().clamp_(lo, hi)
    return out


class _linear_fxp_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, rshift_i, rshift_w, rshift_o, max_abs_i, max_abs_w,
                full_signed_range=False):
        ctx.save_for_backward(input, weight, bias)
        bot_i = -max_abs_i if full_signed_range else 1 - max_abs_i
        top_i = max_abs_i - 1
        i_round = _shift_round_clamp(input, rshift_i, bot_i, top_i)
        bot_w = -max_abs_w if full_signed_range else 1 - max_abs_w
        top_w = max_abs_w - 1
        w_round = _shift_round_clamp(weight, rshift_w, bot_w, top_w)
        output = torch.matmul(i_round, w_round.t())
        output = pow2_rshift(output, rshift_o)
        if bias is not None:
            output = output + bias
        return output


    @staticmethod
    def backward(ctx, grad_output):
        return _linear_ste_grads(ctx, grad_output) + (None,) * (len(ctx.needs_input_grad) - 3)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """Spatial (H, W) of a conv2d output."""
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])
    h = math.floor((h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return h, w


def conv2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
    """Padding (as (top,bottom),(left,right)) to map h_w_in to h_w_out."""
    h_w_in, h_w_out, kernel_size, stride, dilation = num2tuple(h_w_in), num2tuple(h_w_out), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation)
    p_h = ((h_w_out[0] - 1) * stride[0] - h_w_in[0] + dilation[0] * (kernel_size[0] - 1) + 1)
    p_w = ((h_w_out[1] - 1) * stride[1] - h_w_in[1] + dilation[1] * (kernel_size[1] - 1) + 1)
    return (math.floor(p_h / 2), math.ceil(p_h / 2)), (math.floor(p_w / 2), math.ceil(p_w / 2))


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
        if tuple(weight_ext.shape) != (out_channels, in_channels, kh, kw):
            message = f'weight_ext shape {tuple(weight_ext.shape)} != {(out_channels, in_channels, kh, kw)}.'
            logger.error(message)
            raise AssertionError(message)
        module.weight.data = weight_ext.clone().type(module.weight.dtype)
    if bias and bias_ext is not None:
        if tuple(bias_ext.shape) != (out_channels,):
            message = f'bias_ext shape {tuple(bias_ext.shape)} != ({out_channels},).'
            logger.error(message)
            raise AssertionError(message)
        module.bias.data = bias_ext.clone().type(module.bias.dtype)


def _conv2d_binary(input, weight, bias, kernel_size, stride, padding, dilation, linear_fn):
    """
    Run a binary-domain conv2d by im2col + a 2D linear kernel + fold: unfold the input to
    patches, apply ``linear_fn(patches_2d, weight_2d)``, which is a binary linear autograd
    Function, then fold the result back to NCHW. Bias is added after folding. The unfold
    and fold are differentiable, so the linear Function's STE gradient flows through to
    weight and input.

    With :math:`u` the im2col patches and :math:`W_{2d}` the flattened kernel,

    .. math::

       y = \\mathrm{fold}\\!\\left(\\mathrm{linear\\_fn}(u, W_{2d})\\right) + b,

    which equals ``conv2d(input, weight) + bias`` whenever ``linear_fn`` is an exact
    matrix product.
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


def _mgu_run_outlasts_ismul(timestep, depth_ismul):
    """
    Whether a run of ``timestep`` timesteps outlasts the flush of the ``depth_ismul``
    shift register inside the MGU gate multiplier. Flushing that register consumes
    ``2 ** depth_ismul`` timesteps, so a run of exactly that length leaves nothing
    behind. Shared by mgu and mgu_hub, which state the same rule in timestep and in
    width units respectively.
    """
    return timestep > 2 ** depth_ismul


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

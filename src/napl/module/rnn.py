import torch
import math
import torch.nn.functional as F

from napl.utils import *
from napl.base import napl_base
from loguru import logger

# NB: operation primitives are imported lazily inside __init__ (not at module top),
# because napl.operation.mul imports napl.module.encoder, so a top-level import here
# creates a circular import when napl.operation is loaded before napl.module.


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


class mgu_hard(napl_base):
    """
    Minimal Gated Unit (MGU) recurrent cell in the binary (float) domain with hard
    activations: sigmoid -> hard sigmoid, tanh -> hard tanh, so every intermediate value
    stays in the legal unary range. Single-shot, no tick; trainable.
    Refs: "Simplified Minimal Gated Unit Variations for RNNs".
    """
    def __init__(self, input_size, hidden_size, bias=True, config={'hard': True}):
        super().__init__(config, [])
        self.input_size, self.hidden_size, self.bias = input_size, hidden_size, bias
        self.hard = config.get('hard', True)
        from napl.operation import sigmoid_hub, tanh_hub
        self.htanh = tanh_hub()        # the explicit hard-tanh clamps (always hard)
        self.fg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()
        _init_mgu_params(self, input_size, hidden_size, bias)

    def forward(self, input, hx=None):
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        # forget gate
        fg_ug_in = torch.cat((hx, input), 1)
        fg_in = self.htanh(F.linear(fg_ug_in, self.weight_f, self.bias_f))
        fg = self.fg_sigmoid(fg_in)
        # new gate
        fg_hx = fg * hx
        ng_ug_in = torch.cat((fg_hx, input), 1)
        ng = self.ng_tanh(F.linear(ng_ug_in, self.weight_n, self.bias_n))
        # output: hy = hardtanh(ng*(1-fg) + fg*hx)
        fg_ng = fg * ng
        return self.htanh(ng - fg_ng + fg_hx)


class mgu_hardfxp(napl_base):
    """
    MGU cell as mgu_hard, but every operand is rounded to fixed point (intwidth, fracwidth)
    before use (quant-aware). Single-shot, no tick; trainable via STE through round_fxp.
    """
    def __init__(self, input_size, hidden_size, bias=True, config={'hard': True, 'intwidth': 3, 'fracwidth': 4}):
        super().__init__(config, [])
        self.input_size, self.hidden_size, self.bias = input_size, hidden_size, bias
        self.hard = config.get('hard', True)
        from napl.operation import sigmoid_hub, tanh_hub, round_fxp
        self.htanh = tanh_hub()
        self.trunc = round_fxp({'intwidth': config.get('intwidth', 3), 'fracwidth': config.get('fracwidth', 4)})
        self.fg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()
        _init_mgu_params(self, input_size, hidden_size, bias)

    def forward(self, input, hx=None):
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        t = self.trunc
        # round_fxp is a pure (deterministic STE) function of its input, so each operand is
        # truncated once and the result reused across consumers (halves the redundant
        # quantizations vs. recomputing t(hx)/t(input)/t(fg)/t(fg_hx)/t(ng) at every use).
        t_hx, t_input = t(hx), t(input)
        fg_ug_in = torch.cat((t_hx, t_input), 1)
        fg_in = self.htanh(F.linear(t(fg_ug_in), t(self.weight_f), t(self.bias_f)))
        fg = self.fg_sigmoid(t(fg_in))
        t_fg = t(fg)
        fg_hx = t_fg * t_hx
        t_fg_hx = t(fg_hx)
        ng_ug_in = torch.cat((t_fg_hx, t_input), 1)
        ng = self.ng_tanh(t(F.linear(t(ng_ug_in), t(self.weight_n), t(self.bias_n))))
        t_ng = t(ng)
        fg_ng = t_fg * t_ng
        return self.htanh(t_ng - t(fg_ng) + t_fg_hx)


class mgu_fsu(napl_base):
    """
    Streaming (FSU) MGU cell, evaluated one timestep at a time. The two gate linears use a
    saturating scale-1 unary adder (which realizes linear + hard tanh in the unary domain);
    the forget-gate hard sigmoid is the scaled add (x+1)/2; fg*hx uses conditional-spike-
    generation multiply (hx is a fixed value), fg*ng uses XNOR multiply, and the output is a
    scale-1 unary add of [ng, 1-fg*ng, fg*hx] (= hard tanh of ng*(1-fg)+fg*hx). hx is the
    fixed hidden value for this streaming run. Inner cell used by mgu_hub. Bipolar,
    rate coding.
    """
    def __init__(self, weight_f, bias_f, weight_n, bias_n, hx_value,
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol', 'width': 12}):
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        from napl.operation import sigmoid_hard, mul_csg, mul_and, add_any
        from napl.module.linear import linear_fsu
        assert self.polarity == 'bipolar', logger.error('mgu_fsu requires bipolar.')

        ts, gen = config['timestep'], config['generator']
        width = config.get('width', 12)
        self.hx_value = hx_value
        # gate linears on distinct weight rng dims; scale=1 makes the adder saturate (hard tanh)
        def lin(w, b, d):
            return linear_fsu(w, b, {'polarity': 'bipolar', 'timestep': ts, 'generator': gen,
                                     'dim': d, 'scale': 1, 'width': width})
        self.fg_ug_tanh = lin(weight_f, bias_f, 3)
        self.ng_ug_tanh = lin(weight_n, bias_n, 5)
        self.fg_sigmoid = sigmoid_hard({'polarity': 'bipolar'})
        self.fg_hx_mul = mul_csg({'polarity': 'bipolar', 'timestep': ts, 'generator': gen})  # fg (spike) * hx (value)
        self.fg_ng_mul = mul_and({'polarity': 'bipolar'})                                     # fg (spike) * ng (spike)
        self.hy_add = add_any({'polarity': 'bipolar', 'scale': 1, 'width': width})

    def reset(self, verbose=False):
        self.timestep_cur = 0
        for m in [self.fg_ug_tanh, self.ng_ug_tanh, self.fg_sigmoid, self.fg_hx_mul, self.fg_ng_mul, self.hy_add]:
            m.reset()

    def forward(self, input_spike, hx_spike):
        self.tick()
        fg_in = self.fg_ug_tanh(torch.cat((hx_spike, input_spike), dim=1))
        fg = self.fg_sigmoid(fg_in)
        fg_hx = self.fg_hx_mul(fg, self.hx_value)
        ng = self.ng_ug_tanh(torch.cat((fg_hx, input_spike), dim=1))
        fg_ng = self.fg_ng_mul(fg, ng)
        fg_ng_inv = 1 - fg_ng.type(torch.int8)
        # feed the pre-reduced 3-operand sum directly (entry=3 matches the stack size along
        # dim=0), avoiding materializing the stacked tensor; the 0/1 spikes sum to <=3 so the
        # integer add is exact and equals torch.sum(stack, 0, dtype=ntype).
        return self.hy_add(ng + fg_ng_inv + fg_hx, dim=None, entry=3)


class mgu_hub(napl_base):
    """
    Hybrid (single-shot from the caller) MGU: internally encodes input and hx into spike
    streams, runs mgu_fsu over 2**width cycles, and decodes the output with the accuracy
    (progressive-error) metric. Corresponds to mgu_hard with hard activations. Weights are
    external.
    """
    def __init__(self, input_size, hidden_size, bias=True,
                 weight_f=None, bias_f=None, weight_n=None, bias_n=None,
                 config={'polarity': 'bipolar', 'width': 8, 'generator': 'sobol'}):
        super().__init__(config, [])
        self.input_size, self.hidden_size, self.bias = input_size, hidden_size, bias
        self.width = config.get('width', 8)
        self.generator = config.get('generator', 'sobol')
        self.weight_f, self.bias_f = weight_f, bias_f
        self.weight_n, self.bias_n = weight_n, bias_n
        # accumulator width for the inner linears must hold the fan-in (hidden+input+bias)
        entry = hidden_size + input_size + (1 if bias else 0)
        self.lin_width = max(12, math.ceil(math.log2(entry)) + 2)

    def forward(self, input, hx=None):
        from napl.module.encoder import encoder
        from napl.metric import accuracy
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        ts = 2 ** self.width

        def enc(d):
            return encoder({'polarity': 'bipolar', 'timestep': ts, 'generator': self.generator, 'dim': d})
        i_enc, h_enc = enc(1), enc(2)
        cell = mgu_fsu(self.weight_f, self.bias_f, self.weight_n, self.bias_n, hx,
                       {'polarity': 'bipolar', 'timestep': ts, 'generator': self.generator, 'width': self.lin_width})
        acc = accuracy({'polarity': 'bipolar'})
        for _ in range(ts):
            acc(cell(i_enc(input), h_enc(hx)))
        return acc.spike_value


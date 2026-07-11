"""
uBrain Cascade_CNN_RNN ported into napl.

uBrain (UnarySim app/uBrain) is a brain-computer-interface accelerator that
classifies EEG with dynamic rate-coded unary computing. The network is a cascade
CNN + RNN:

    conv1 (1 -> cnn_chn, 3x3, pad 1) -> ScaleReLU
    conv2 (cnn_chn -> 2*cnn_chn, 3x3, pad 1) -> ScaleReLU
    flatten -> fc3 (-> fc_sz) -> ScaleReLU -> dropout
    reshape to (win, fc_sz) -> MGU recurrent cell looped over `win` steps
    fc5 (rnn_hidden_sz -> sum(num_class)) -> Hardtanh
    -> split into the per-head logits (default two heads [5, 2])

This module provides two faithful forms of that network built from napl kernels:

    Cascade_CNN_RNN_FP   nn.Conv2d / nn.Linear + mgu_hard + relu_hub/tanh_hub
                         (float domain, trainable; == UnarySim model_fp).
    Cascade_CNN_RNN_HUB  conv_hub / linear_hub + mgu_hub + relu_hub/tanh_hub
                         (hybrid unary-binary inference; == UnarySim model_hub).

The HUB form takes its weights from an FP instance (build_hub_from_fp), so the
two share identical parameters and the HUB output can be compared against the FP
output as a fidelity check.

UnarySim -> napl kernel map used here:
    nn.Conv2d / nn.Linear   -> conv_hub / linear_hub   (HUB inference path)
    HardMGUCell             -> mgu_hard (FP) / mgu_hub (HUB)
    ScaleReLU               -> relu_hub
    nn.Hardtanh             -> tanh_hub
    truncated_normal        -> napl.utils.truncated_normal

No EEG data, checkpoint, or hardware is involved: this is the model itself, run
on EEG-shaped tensors.
"""

import torch
import torch.nn as nn

from napl.utils import truncated_normal
from napl.module import conv_hub, linear_hub, mgu_hard, mgu_hub
from napl.operation import relu_hub, tanh_hub


def fc3_in_features(input_sz, cnn_chn, cnn_padding):
    """Flatten size feeding fc3 (matches the upstream same-conv formula)."""
    h = input_sz[0] + 2 * 2 * (cnn_padding - 1)
    w = input_sz[1] + 2 * 2 * (cnn_padding - 1)
    return h * w * cnn_chn * 2


class Cascade_CNN_RNN_FP(nn.Module):
    """
    FP (float-domain) uBrain network. Trainable; the napl equivalent of
    UnarySim app/uBrain/model/model_fp.py with linear_act='scalerelu'.
    """

    def __init__(self,
                 input_sz=(10, 11),       # 10-10 MI grid
                 cnn_chn=16,
                 cnn_kn_sz=3,
                 cnn_padding=1,
                 fc_sz=256,
                 rnn_win_sz=10,
                 rnn_hidden_sz=64,
                 bias=False,
                 keep_prob=0.5,
                 num_class=(5, 2),
                 init_std=0.05):
        super().__init__()
        self.input_sz = tuple(input_sz)
        self.fc_sz = fc_sz
        self.rnn_win_sz = rnn_win_sz
        self.rnn_hidden_sz = rnn_hidden_sz
        self.num_class = tuple(num_class)
        self.cnn_chn = cnn_chn
        self.cnn_padding = cnn_padding
        self.bias = bias

        self.conv1 = nn.Conv2d(1, cnn_chn, (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding)
        self.conv2 = nn.Conv2d(cnn_chn, cnn_chn * 2, (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding)
        self.fc3 = nn.Linear(fc3_in_features(self.input_sz, cnn_chn, cnn_padding), fc_sz, bias=bias)
        self.fc3_drop = nn.Dropout(p=1 - keep_prob)
        # MGU recurrent cell, hard activations (== HardMGUCell)
        self.rnncell4 = mgu_hard(fc_sz, rnn_hidden_sz, bias=bias, config={'hard': True})
        self.fc5 = nn.Linear(rnn_hidden_sz, sum(num_class), bias=bias)

        # ScaleReLU == relu_hub (Hardtanh(0, 1)); output Hardtanh == tanh_hub
        self.conv1_act = relu_hub({'scale': 1.0})
        self.conv2_act = relu_hub({'scale': 1.0})
        self.fc3_act = relu_hub({'scale': 1.0})
        self.out_act = tanh_hub()

        self._init_weight(init_std)

    def _init_weight(self, std):
        self.conv1.weight.data = truncated_normal(self.conv1.weight, 0.0, std)
        self.conv2.weight.data = truncated_normal(self.conv2.weight, 0.0, std)
        self.fc3.weight.data = truncated_normal(self.fc3.weight, 0.0, std)
        self.fc5.weight.data = truncated_normal(self.fc5.weight, 0.0, std)

    def forward(self, x):
        # x: (batch, win, h, w)
        o = x.view(-1, 1, self.input_sz[0], self.input_sz[1])      # (batch*win, 1, h, w)
        o = self.conv1_act(self.conv1(o))
        o = self.conv2_act(self.conv2(o))
        o = o.view(o.shape[0], -1)
        o = self.fc3_act(self.fc3(o))
        o = self.fc3_drop(o)
        o = o.view(-1, self.rnn_win_sz, self.fc_sz).transpose(0, 1)  # (win, batch, fc_sz)

        hx = torch.zeros(o[0].size(0), self.rnn_hidden_sz, dtype=x.dtype, device=x.device)
        for i in range(self.rnn_win_sz):
            hx = self.rnncell4(o[i], hx)

        out = self.out_act(self.fc5(hx))
        return out                                                  # (batch, sum(num_class))

    def split_heads(self, out):
        """Split the flat output into per-head logits."""
        return list(torch.split(out, list(self.num_class), dim=1))


class Cascade_CNN_RNN_HUB(nn.Module):
    """
    HUB (hybrid unary-binary) uBrain network. Inference only; weights come from an
    FP instance via build_hub_from_fp. The napl equivalent of UnarySim
    app/uBrain/model/model_hub.py.

    `width` is the unary bitwidth: conv/linear use cycle=2**(width-1) and the MGU
    streams 2**width cycles per call (so keep width modest; the MGU dominates cost).
    """

    def __init__(self,
                 input_sz=(10, 11),
                 cnn_chn=16,
                 cnn_kn_sz=3,
                 cnn_padding=1,
                 fc_sz=256,
                 rnn_win_sz=10,
                 rnn_hidden_sz=64,
                 bias=False,
                 keep_prob=0.5,
                 num_class=(5, 2),
                 width=8,
                 rng='sobol',
                 conv1_weight=None, conv2_weight=None,
                 fc3_weight=None, fc5_weight=None,
                 rnn4_weight_f=None, rnn4_bias_f=None,
                 rnn4_weight_n=None, rnn4_bias_n=None):
        super().__init__()
        assert not bias, 'HUB port wired for bias=False (the uBrain default).'
        self.input_sz = tuple(input_sz)
        self.fc_sz = fc_sz
        self.rnn_win_sz = rnn_win_sz
        self.rnn_hidden_sz = rnn_hidden_sz
        self.num_class = tuple(num_class)
        self.width = width
        cycle = 2 ** (width - 1)

        # widthi == widthw required by the HUB value map; cycle = 2**(width-1)
        lin_cfg = lambda: {'widthi': width, 'rngi': rng, 'quantilei': 1,
                           'widthw': width, 'rngw': rng, 'quantilew': 1,
                           'cycle': cycle, 'rounding': 'round'}

        self.conv1 = conv_hub(1, cnn_chn, (cnn_kn_sz, cnn_kn_sz), padding=cnn_padding,
                              bias=False, weight_ext=conv1_weight, config=lin_cfg())
        self.conv2 = conv_hub(cnn_chn, cnn_chn * 2, (cnn_kn_sz, cnn_kn_sz), padding=cnn_padding,
                              bias=False, weight_ext=conv2_weight, config=lin_cfg())
        self.fc3 = linear_hub(fc3_in_features(self.input_sz, cnn_chn, cnn_padding), fc_sz,
                              bias=False, weight_ext=fc3_weight, config=lin_cfg())
        self.fc3_drop = nn.Dropout(p=1 - keep_prob)
        self.rnncell4 = mgu_hub(fc_sz, rnn_hidden_sz, bias=False,
                                weight_f=rnn4_weight_f, bias_f=rnn4_bias_f,
                                weight_n=rnn4_weight_n, bias_n=rnn4_bias_n,
                                config={'polarity': 'bipolar', 'width': width, 'generator': rng})
        self.fc5 = linear_hub(rnn_hidden_sz, sum(num_class), bias=False,
                              weight_ext=fc5_weight, config=lin_cfg())

        self.conv1_act = relu_hub({'scale': 1.0})
        self.conv2_act = relu_hub({'scale': 1.0})
        self.fc3_act = relu_hub({'scale': 1.0})
        self.out_act = tanh_hub()

    def forward(self, x):
        # x: (batch, win, h, w)
        o = x.view(-1, 1, self.input_sz[0], self.input_sz[1])
        o = self.conv1_act(self.conv1(o))
        o = self.conv2_act(self.conv2(o))
        o = o.view(o.shape[0], -1)
        o = self.fc3_act(self.fc3(o))
        o = self.fc3_drop(o)
        o = o.view(-1, self.rnn_win_sz, self.fc_sz).transpose(0, 1)  # (win, batch, fc_sz)

        hx = torch.zeros(o[0].size(0), self.rnn_hidden_sz, dtype=x.dtype, device=x.device)
        for i in range(self.rnn_win_sz):
            hx = self.rnncell4(o[i], hx)

        out = self.out_act(self.fc5(hx))
        return out

    def split_heads(self, out):
        return list(torch.split(out, list(self.num_class), dim=1))


def build_hub_from_fp(fp_model, width=8, rng='sobol'):
    """
    Build a HUB Cascade_CNN_RNN that shares the FP model's weights. The conv/linear
    nn weights map straight into conv_hub/linear_hub weight_ext (identical shapes),
    and the mgu_hard gate weights (shape (hidden, hidden+input)) map into mgu_hub's
    weight_f / weight_n.
    """
    cnn_chn = fp_model.cnn_chn
    return Cascade_CNN_RNN_HUB(
        input_sz=fp_model.input_sz,
        cnn_chn=cnn_chn,
        cnn_kn_sz=fp_model.conv1.kernel_size[0],
        cnn_padding=fp_model.cnn_padding,
        fc_sz=fp_model.fc_sz,
        rnn_win_sz=fp_model.rnn_win_sz,
        rnn_hidden_sz=fp_model.rnn_hidden_sz,
        bias=fp_model.bias,
        num_class=fp_model.num_class,
        width=width,
        rng=rng,
        conv1_weight=fp_model.conv1.weight.detach().clone(),
        conv2_weight=fp_model.conv2.weight.detach().clone(),
        fc3_weight=fp_model.fc3.weight.detach().clone(),
        fc5_weight=fp_model.fc5.weight.detach().clone(),
        rnn4_weight_f=fp_model.rnncell4.weight_f.detach().clone(),
        rnn4_bias_f=None,
        rnn4_weight_n=fp_model.rnncell4.weight_n.detach().clone(),
        rnn4_bias_n=None,
    )

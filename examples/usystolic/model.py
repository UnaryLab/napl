"""
The uSystolic convnet_mnist network in three forms:

  - ConvNetFP   : the FP32 baseline (nn.Conv2d / nn.Linear), trained by train_fp.py.
  - ConvNetHUB  : the uSystolic unary-MAC inference model (conv_hub / linear_hub),
                  knob = `cycle` (unary MAC cycle count), no retraining.
  - ConvNetFXP  : the fixed-point inference model (conv_fxp / linear_fxp),
                  knob = `bitwidth`, with cycle = 2**(bitwidth-1).

The architecture matches UnarySim app/uSystolic/convnet_mnist exactly:
  conv1(1->32,3) relu  conv2(32->64,3) relu  maxpool(2)  dropout(0.25)
  flatten  fc1(9216->128) relu  dropout(0.5)  fc2(128->10)  log_softmax

napl has no pooling op, so max-pooling stays stock torch F.max_pool2d. This is
faithful to how UnarySim composes the layers: the HUB/FXP conv/linear cells are
single-shot binary-domain modules, and pooling sits between them unchanged in the
upstream code too. Dropout is identity at eval (model.eval()), so it is harmless
in the HUB/FXP models and kept only for structural parity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from napl.module import conv_hub, linear_hub, conv_fxp, linear_fxp


class ConvNetFP(nn.Module):
    """FP32 baseline LeNet-ish CNN (trained by train_fp.py)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _state_to_tensors(state_dict):
    """Pull the FP conv/linear weight+bias tensors out of a ConvNetFP state_dict, in order."""
    return (
        state_dict["conv1.weight"], state_dict["conv1.bias"],
        state_dict["conv2.weight"], state_dict["conv2.bias"],
        state_dict["fc1.weight"], state_dict["fc1.bias"],
        state_dict["fc2.weight"], state_dict["fc2.bias"],
    )


class ConvNetHUB(nn.Module):
    """
    uSystolic unary-MAC inference model: conv/linear layers are HUB cells whose
    products come from the unary-multiplication value map. `cycle` is the unary MAC
    cycle count; weights/biases are loaded from the FP checkpoint (no retraining).
    """

    def __init__(self, state_dict, cycle, width=8, rng="sobol"):
        super().__init__()
        w = _state_to_tensors(state_dict)
        cfg = {"widthi": width, "rngi": rng, "quantilei": 1,
               "widthw": width, "rngw": rng, "quantilew": 1,
               "cycle": cycle, "rounding": "round"}
        self.conv1 = conv_hub(1, 32, 3, 1, weight_ext=w[0], bias_ext=w[1], config=cfg)
        self.conv2 = conv_hub(32, 64, 3, 1, weight_ext=w[2], bias_ext=w[3], config=cfg)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = linear_hub(9216, 128, weight_ext=w[4], bias_ext=w[5], config=cfg)
        self.fc2 = linear_hub(128, 10, weight_ext=w[6], bias_ext=w[7], config=cfg)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ConvNetFXP(nn.Module):
    """
    Fixed-point inference model: conv/linear layers are FXP cells. `bitwidth` sets
    widthi == widthw == bitwidth (the upstream keep_res="input" / i-res config; napl
    has no o-res split). Weights/biases come from the FP checkpoint (no retraining).
    """

    def __init__(self, state_dict, bitwidth):
        super().__init__()
        w = _state_to_tensors(state_dict)
        cfg = {"widthi": bitwidth, "quantilei": 1,
               "widthw": bitwidth, "quantilew": 1, "rounding": "round"}
        self.conv1 = conv_fxp(1, 32, 3, 1, weight_ext=w[0], bias_ext=w[1], config=cfg)
        self.conv2 = conv_fxp(32, 64, 3, 1, weight_ext=w[2], bias_ext=w[3], config=cfg)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = linear_fxp(9216, 128, weight_ext=w[4], bias_ext=w[5], config=cfg)
        self.fc2 = linear_fxp(128, 10, weight_ext=w[6], bias_ext=w[7], config=cfg)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

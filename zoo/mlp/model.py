"""
MLP3 model for the napl MNIST unary-computing example.

Ported from UnarySim app/mlp/model.py (classes MLP3 and MLP3_clamp_eval). Depends on
torch only, no UnarySim. The input size is configurable; the default fan-in is 1024
(a 32x32 MNIST image flattened). The clamp-eval variant clamps each layer's pre-activation
to [-1, 1] so the activations stay in the bipolar stochastic-computing range, which is what
the unary pipeline in eval_unary.py reproduces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP3(nn.Module):
    """3-layer MLP: in_size -> width -> width -> 10. Trained in floating point."""

    def __init__(self, in_size=32 * 32, width=512, p=0.5):
        super().__init__()
        self.in_size = in_size
        self.fc1 = nn.Linear(in_size, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 10)

        self.do1 = nn.Dropout(p=p)
        self.do2 = nn.Dropout(p=p)

    def forward(self, x):
        x = x.view(-1, self.in_size)
        x = self.fc1(x)
        x = F.relu(self.do1(x))
        x = self.fc2(x)
        x = F.relu(self.do2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class MLP3_clamp_train(nn.Module):
    """
    Training variant of MLP3 that clamps each activation to [-1, 1] right after ReLU, so the
    learned weights keep the pre-activations inside the bipolar [-1, 1] range that the unary
    stochastic-computing datapath can represent (a unary bipolar stream saturates at +-1).
    Training in this regime is what lets the unary inference in eval_unary.py track the FP
    baseline instead of collapsing to chance. Ported from UnarySim MLP3_clamp_train.
    """

    def __init__(self, in_size=32 * 32, width=512, p=0.2):
        super().__init__()
        self.in_size = in_size
        self.fc1 = nn.Linear(in_size, width)
        self.fc1_drop = nn.Dropout(p)
        self.fc2 = nn.Linear(width, width)
        self.fc2_drop = nn.Dropout(p)
        self.fc3 = nn.Linear(width, 10)

    def forward(self, x):
        x = x.view(-1, self.in_size)
        x = F.relu(self.fc1(x)).clamp(-1, 1)
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x)).clamp(-1, 1)
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)


class MLP3_clamp_eval(nn.Module):
    """
    Evaluation variant of MLP3 with each pre-activation clamped to [-1, 1] before ReLU,
    so weights, biases and activations all live in the bipolar [-1, 1] range that the
    unary stochastic-computing pipeline operates in. Exposes the per-layer outputs
    (fc1_out, relu1_out, ...) as attributes so they can be used as references.
    """

    def __init__(self, in_size=32 * 32, width=512):
        super().__init__()
        self.in_size = in_size
        self.fc1 = nn.Linear(in_size, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 10)

        self.fc1_out = torch.zeros(1)
        self.relu1_out = torch.zeros(1)
        self.fc2_out = torch.zeros(1)
        self.relu2_out = torch.zeros(1)
        self.fc3_out = torch.zeros(1)

    def forward(self, x):
        x = x.view(-1, self.in_size)
        self.fc1_out = self.fc1(x).clamp(-1, 1)
        self.relu1_out = F.relu(self.fc1_out)
        self.fc2_out = self.fc2(self.relu1_out).clamp(-1, 1)
        self.relu2_out = F.relu(self.fc2_out)
        self.fc3_out = self.fc3(self.relu2_out).clamp(-1, 1)
        return F.softmax(self.fc3_out, dim=1)

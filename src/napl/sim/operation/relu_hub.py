import torch

from napl.sim.base import napl_base


class relu_hub(napl_base):
    """
    Binary-domain ReLU: clip the input to [0, scale] (a Hardtanh). Single-shot;
    trainable for binary-domain quant-aware training.
    """
    streaming = False
    def __init__(
            self,
            config={
                'scale': 1.0,
            }
        ):
        super().__init__(config, [])
        self.scale = config.get('scale', 1.0)

    def forward(self, input):
        return torch.nn.functional.hardtanh(input, 0.0, self.scale)

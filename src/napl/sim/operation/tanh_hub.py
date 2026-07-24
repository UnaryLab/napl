import torch

from napl.sim.base import napl_base


class tanh_hub(napl_base):
    """
    Binary-domain hard tanh: clip the input to [-1, 1] (a Hardtanh), the binary-domain
    counterpart used for training/inference. Single-shot.
    """
    streaming = False
    def __init__(
        self,
        config={},
    ):
        super().__init__(config, [])
        self.delay = 0

    def forward(self, input: torch.tensor):
        return torch.nn.functional.hardtanh(input, -1.0, 1.0)

import torch

from napl.sim.base import napl_base


class sigmoid_hub(napl_base):
    """
    Binary-domain hard sigmoid: Hardsigmoid(input * scale), a piecewise-linear
    approximation of the sigmoid. Single-shot. Default scale 3.
    """
    streaming = False
    def __init__(
        self,
        config={
            'scale': 3,
        },
    ):
        super().__init__(config, [])
        self.delay = 0
        self.scale = config.get('scale', 3)

    def forward(self, input: torch.tensor):
        return torch.nn.functional.hardsigmoid(input * self.scale)

import torch

from napl.sim.operation import round_ste
from napl.utils._shared_test import devices


INTWIDTH = 3
FRACWIDTH = 4
MIN_CODE = 1 - 2 ** (INTWIDTH + FRACWIDTH)
MAX_CODE = 2 ** (INTWIDTH + FRACWIDTH) - 1


class round_reference(torch.nn.Module):
    def forward(self, input):
        return (
            torch.round(input * 2**FRACWIDTH)
            .clamp(MIN_CODE, MAX_CODE)
            / 2**FRACWIDTH
        )


def test_round_ste_dtype_and_boundaries():
    for device in devices():
        input = torch.tensor(
            [-100.0, -0.3, 0.1, 100.0],
            dtype=torch.float32,
            device=device,
        )
        result = round_ste(
            input,
            fracwidth=FRACWIDTH,
            min_val=MIN_CODE,
            max_val=MAX_CODE,
        )
        expected = round_reference().to(device)(input)
        assert result.dtype == input.dtype
        assert torch.equal(result, expected)


if __name__ == '__main__':
    test_round_ste_dtype_and_boundaries()

import itertools
from pathlib import Path

import torch

from napl.sim.operation import mul_gaines


VEC = Path(__file__).resolve().parent.parent / "vec" / "mul_gaines.vec"


def main():
    unipolar = mul_gaines({"polarity": "unipolar"})
    bipolar = mul_gaines({"polarity": "bipolar"})
    VEC.parent.mkdir(parents=True, exist_ok=True)

    with VEC.open("w") as output:
        for input_0, input_1 in itertools.product((0, 1), repeat=2):
            operands = (torch.tensor(input_0), torch.tensor(input_1))
            out_uni = int(unipolar(*operands).item())
            out_bi = int(bipolar(*operands).item())
            output.write(
                f"{input_0} {input_1} {out_uni} {out_bi}\n"
            )

    print(f"wrote {VEC} (4 vectors)")


if __name__ == "__main__":
    main()

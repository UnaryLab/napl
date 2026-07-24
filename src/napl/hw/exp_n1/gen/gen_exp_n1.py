import sys
from pathlib import Path

import torch

from napl.sim.operation import exp_n1


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "exp_n1.vec"
PARAMS = ROOT / "vec" / "exp_n1_params.vh"
ROM = ROOT / "vec" / "exp_n1_rom.hex"
CONFIG = {
    "polarity": "unipolar",
    "timestep": 256,
    "generator": "sobol",
    "dim": 1,
}
CODEC = {
    "polarity": "unipolar",
    "timestep": 256,
    "generator": "sobol",
    "dim": 5,
}


def main():
    stream = []
    for value in rep_values("unipolar"):
        stream.extend(encode_value(CODEC, value))

    model = exp_n1(dict(CONFIG))
    model.reset()
    reset_at = len(stream) // 2
    VEC.parent.mkdir(parents=True, exist_ok=True)
    ROM.write_text(
        "\n".join(
            "".join(str(bit) for bit in bits)
            for bits in model._const_bits
        )
        + "\n"
    )
    PARAMS.write_text(
        f"`define GEN_WIDTH {model.width}\n"
        f"`define GEN_PP_DELAY {model.hw.pp_delay}\n"
    )

    with VEC.open("w") as output:
        for index, bit in enumerate(stream):
            reset_flag = int(index == 0 or index == reset_at)
            if index == reset_at:
                model.reset()
            input_spike = torch.tensor([bit], dtype=model.stype)
            result = int(model(input_spike).item())
            output.write(f"{reset_flag} {bit} {result}\n")

    print(
        f"wrote {VEC} ({len(stream)} vectors, reset@{reset_at}), "
        f"{PARAMS}, and {ROM} ({len(model._const_bits)} ROM lines)"
    )


if __name__ == "__main__":
    main()

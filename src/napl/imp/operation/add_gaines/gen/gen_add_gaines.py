import math
from pathlib import Path

import torch

from napl.sim.operation import add_gaines


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "add_gaines.vec"
PARAMS = ROOT / "vec" / "add_gaines_params.vh"
ROM = ROOT / "vec" / "add_gaines_rom.hex"
ENTRY = 8
SELECT_WIDTH = int(math.log2(ENTRY))
SEGMENT = list(range(1 << ENTRY))
SCALED = {
    "scaled": True,
    "entry": ENTRY,
    "generator": "sobol",
    "dim": 5,
}
UNSCALED = {"scaled": False}


def bits(value):
    return [(value >> index) & 1 for index in range(ENTRY)]


def bus(value):
    return f"{value:0{ENTRY}b}"


def main():
    scaled_uni = add_gaines({"polarity": "unipolar", **SCALED})
    scaled_bi = add_gaines({"polarity": "bipolar", **SCALED})
    unscaled = add_gaines({"polarity": "unipolar", **UNSCALED})
    models = (scaled_uni, scaled_bi, unscaled)
    rows = SEGMENT + SEGMENT
    reset_at = len(SEGMENT)

    VEC.parent.mkdir(parents=True, exist_ok=True)
    ROM.write_text(
        "\n".join(
            f"{value:0{SELECT_WIDTH}b}" for value in scaled_uni.sel_seq
        )
        + "\n"
    )
    PARAMS.write_text(
        f"`define GEN_SCALED {int(SCALED['scaled'])}\n"
        f"`define GEN_UNSCALED {int(UNSCALED['scaled'])}\n"
        f"`define GEN_ENTRY {ENTRY}\n"
        f"`define GEN_SELECT_WIDTH {SELECT_WIDTH}\n"
        f"`define GEN_PP_DELAY {scaled_uni.hw.pp_delay}\n"
    )

    with VEC.open("w") as output:
        for index, value in enumerate(rows):
            reset = int(index == 0 or index == reset_at)
            if reset:
                for model in models:
                    model.reset()

            value_bi = value ^ 0xA5
            value_or = (value * 37) & 0xFF
            in_uni = torch.tensor(bits(value), dtype=scaled_uni.stype)
            in_bi = torch.tensor(bits(value_bi), dtype=scaled_bi.stype)
            in_or = torch.tensor(bits(value_or), dtype=unscaled.stype)
            out_uni = int(scaled_uni(in_uni, dim=0).item())
            out_bi = int(scaled_bi(in_bi, dim=0).item())
            out_or = int(unscaled(in_or, dim=0).item())
            output.write(
                f"{reset} {bus(value)} {out_uni} "
                f"{bus(value_bi)} {out_bi} {bus(value_or)} {out_or}\n"
            )

    print(
        f"wrote {VEC} ({len(rows)} vectors, reset@{reset_at}), "
        f"{PARAMS}, and {ROM} ({len(scaled_uni.sel_seq)} ROM lines)"
    )


if __name__ == "__main__":
    main()

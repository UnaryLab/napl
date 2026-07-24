import sys
from pathlib import Path

import torch

from napl.sim.operation import exp_ng


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "exp_ng.vec"
PARAMS = ROOT / "vec" / "exp_ng_params.vh"
CONFIG = {"depth": 5, "gain": 1}
CODEC = {
    "polarity": "bipolar",
    "timestep": 256,
    "generator": "sobol",
    "dim": 1,
}


def main():
    stream = []
    for value in rep_values("unipolar", value_range=(0.0, 1.0)):
        stream.extend(encode_value(CODEC, value))

    model = exp_ng(dict(CONFIG))
    model.reset()
    reset_at = len(stream) // 2
    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_DEPTH {CONFIG['depth']}\n"
        f"`define GEN_GAIN {CONFIG['gain']}\n"
        f"`define GEN_PP_DELAY {model.hw.pp_delay}\n"
    )

    with VEC.open("w") as output:
        for index, bit in enumerate(stream):
            if index == reset_at:
                model.reset()
                output.write("R\n")
            result = int(model(torch.tensor(bit, dtype=model.stype)).item())
            output.write(f"{bit} {result}\n")

    print(
        f"wrote {VEC} ({len(stream)} vectors, reset@{reset_at}) "
        f"and {PARAMS}"
    )


if __name__ == "__main__":
    main()

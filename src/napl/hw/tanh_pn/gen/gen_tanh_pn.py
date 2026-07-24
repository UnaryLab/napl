import sys
from pathlib import Path

import torch

from napl.sim.operation import tanh_pn


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values


ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "tanh_pn.vec"
PARAMS = ROOT / "vec" / "tanh_pn_params.vh"
CONFIG = {"depth": 3}
CODEC = {
    "polarity": "bipolar",
    "timestep": 256,
    "generator": "sobol",
    "dim": 1,
}


def main():
    stream = []
    for value in rep_values("bipolar"):
        stream.extend(encode_value(CODEC, value))

    model = tanh_pn(dict(CONFIG))
    model.reset()
    reset_at = len(stream) // 2
    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_DEPTH {CONFIG['depth']}\n"
        f"`define GEN_PP_DELAY {model.hw.pp_delay}\n"
    )

    with VEC.open("w") as output:
        for index, bit in enumerate(stream):
            if index == reset_at:
                model.reset()
                output.write("R\n")
            input_spike = torch.tensor([bit], dtype=model.stype)
            result = int(model(input_spike).item())
            output.write(f"{bit} {result}\n")

    print(
        f"wrote {VEC} ({len(stream)} vectors, reset@{reset_at}) "
        f"and {PARAMS}"
    )


if __name__ == "__main__":
    main()

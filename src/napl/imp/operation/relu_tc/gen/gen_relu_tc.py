"""Generate relu_tc vectors from the NAPL Python model."""
import sys
from pathlib import Path

import torch

from napl.sim.operation import relu_tc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "relu_tc.vec"
PARAMS = ROOT / "vec" / "relu_tc_params.vh"

# Mirrors tests/operation/test_relu_tc.py, including its temporal encoder.
CONFIG = {"width": 8}
CODEC = {
    "polarity": "bipolar",
    "timestep": 256,
    "generator": "temporal",
    "dim": 1,
}


def build_stream():
    stream = []
    for value in rep_values(CODEC["polarity"]):
        stream.extend(encode_value(CODEC, value))
    return stream


def main():
    model = relu_tc(dict(CONFIG))
    model.reset()
    stream = build_stream()
    reset_at = len(stream) // 2

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_WIDTH {CONFIG['width']}\n"
        f"`define GEN_PP_DELAY {model.hw.pp_delay}\n"
    )

    rows = 0
    with VEC.open("w") as output:
        for index, bit in enumerate(stream):
            reset_flag = int(index == 0 or index == reset_at)
            if reset_flag:
                model.reset()
            result = model(torch.tensor(bit, dtype=model.stype))
            output.write(f"{reset_flag} {bit} {int(result.item())}\n")
            rows += 1

    print(f"wrote {VEC} ({rows} vectors, reset@0/{reset_at}) and {PARAMS}")


if __name__ == "__main__":
    main()

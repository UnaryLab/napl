"""Generate relu_delay vectors from the NAPL Python model."""
import sys
from pathlib import Path

import torch

from napl.sim.operation import relu_delay

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

ROOT = Path(__file__).resolve().parent.parent
VEC = ROOT / "vec" / "relu_delay.vec"
PARAMS = ROOT / "vec" / "relu_delay_params.vh"

# Configuration mirrors test_relu_delay.py.
CONFIG = {"depth": 8}
CODEC = {
    "polarity": "bipolar",
    "timestep": 256,
    "generator": "sobol",
    "dim": 1,
}


def build_stream():
    stream = []
    for value in rep_values(CODEC["polarity"]):
        stream.extend(encode_value(CODEC, value))
    return stream


def main():
    model = relu_delay(dict(CONFIG))
    model.reset()
    stream = build_stream()
    reset_at = len(stream) // 2

    VEC.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with VEC.open("w") as output:
        for index, bit in enumerate(stream):
            reset_flag = int(index == 0 or index == reset_at)
            if reset_flag:
                model.reset()
            result = model(torch.tensor(bit, dtype=model.stype))
            output.write(f"{reset_flag} {bit} {int(result.item())}\n")
            rows += 1

        # Fill the delay line with ones so count reaches its top value DEPTH, then
        # drain it with zeros so the outputs read that count back.
        peak = 0
        for bit in [1] * (2 * CONFIG["depth"]) + [0] * (2 * CONFIG["depth"]):
            result = model(torch.tensor(bit, dtype=model.stype))
            peak = max(peak, int(model.count.max().item()))
            output.write(f"0 {bit} {int(result.item())}\n")
            rows += 1
        assert peak == CONFIG["depth"], f"fill block reached count {peak}, not {CONFIG['depth']}"

    PARAMS.write_text(
        f"`define GEN_DEPTH {CONFIG['depth']}\n"
        f"`define GEN_PP_DELAY {model.hw.pp_delay}\n"
        f"`define GEN_VECTORS {rows}\n"
    )

    print(f"wrote {VEC} ({rows} vectors, reset@0/{reset_at}) and {PARAMS}")


if __name__ == "__main__":
    main()

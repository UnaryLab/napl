"""
Generate golden test vectors for the bi2uni RTL module straight from napl's
functional Python model (napl.sim.operation.bi2uni) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

Output: ../vec/bi2uni.vec, one line per cycle:

    <i_input> <o_out>        (each 0/1, space-separated)

bi2uni is stateful (a signed accumulator), so a single input stream is driven
through the model cycle by cycle from reset() and the per-cycle (input, output)
pair is recorded. The input spikes are the SAME ones tests/operation/test_bi2uni.py
sends the op: representative nonnegative operand values encoded by the test's
encoder config (codec_config1: bipolar / timestep=256 / sobol / dim=1). Each
operand starts from reset, and the first stream is also reset after a prefix that
dirties the accumulator.

The sizing param WIDTH is the single source of truth here: it is read from the
op config (mirroring test_bi2uni.py's bi2uni_config), used to build the model,
AND emitted into ../vec/bi2uni_params.vh as `GEN_WIDTH so the testbench overrides
the RTL parameter with the same value. RTL and sim therefore inherit WIDTH from
one place; they cannot drift.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_bi2uni.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import bi2uni

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "bi2uni.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "bi2uni_params.vh"

# Sizing and encoder settings mirror test_bi2uni.py.
BI2UNI = {"width": 2}
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}

# R is outside the spike alphabet and requests a matching model/RTL reset.
RST = "R"


def build_segments():
    """Independent test-derived streams plus one dirtied-state reset."""
    values = rep_values(CODEC["polarity"], value_range=(0.0, 1.0))
    streams = [encode_value(CODEC, value) for value in values]
    return [streams[0][:2], streams[0][2:], *streams[1:]]


def main():
    model = bi2uni({"width": BI2UNI["width"]})

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(f"`define GEN_WIDTH {BI2UNI['width']}\n")

    rows = 0
    with VEC.open("w") as f:
        for seg_idx, seg in enumerate(build_segments()):
            if seg_idx > 0:
                f.write(f"{RST} {RST}\n")
                rows += 1
            model.reset()
            for bit in seg:
                i_input = torch.tensor([bit], dtype=model.stype)
                o_out = int(model(i_input).item())
                f.write(f"{bit} {o_out}\n")
                rows += 1
    print(f"wrote {VEC} ({rows} vectors) and {PARAMS} (GEN_WIDTH={BI2UNI['width']})")


if __name__ == "__main__":
    main()

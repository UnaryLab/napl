"""
Generate golden test vectors for the shiftreg RTL module straight from napl's
functional Python model (napl.sim.operation.shiftreg) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

shiftreg is stateful: a depth-DEPTH bit-serial delay line whose reset() state is
the alternating pattern reg[i] = i % 2 (NOT all-zeros). We drive the model from
reset() with a known input stream and record (input, output) every cycle. The
output at cycle t is what forward() returns at that timestep: the oldest cell,
read BEFORE the new input is pushed.

Output: ../vec/shiftreg.vec, one line per cycle:

    <in> <out>        (each 0/1, space-separated)

The sizing param DEPTH is the single source of truth here: it is read from the
op config (mirroring test_shiftreg.py's shiftreg_config), used to build the
model, AND emitted into ../vec/shiftreg_params.vh as `GEN_DEPTH so the testbench
overrides the RTL parameter with the same value. RTL and sim therefore inherit
DEPTH from one place; they cannot drift.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_shiftreg.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import shiftreg

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "shiftreg.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "shiftreg_params.vh"

# test_shiftreg.py shiftreg_config: the sizing param the op is built with.
SHIFTREG = {"depth": 2}
# test_shiftreg.py codec_config: the encoder feeding shiftreg.
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def main():
    model = shiftreg(config=SHIFTREG)
    model.reset()

    # the test's encoder streams for representative operands, concatenated.
    stream = []
    for v in rep_values(CODEC["polarity"]):
        stream += encode_value(CODEC, v)

    VEC.parent.mkdir(parents=True, exist_ok=True)
    # Emit the param header the testbench includes to override the RTL parameter.
    PARAMS.write_text(f"`define GEN_DEPTH {SHIFTREG['depth']}\n")

    rows = 0
    with VEC.open("w") as f:
        for bit in stream:
            inp = torch.tensor(bit, dtype=model.stype)
            out = int(model(inp).item())
            f.write(f"{bit} {out}\n")
            rows += 1
    print(f"wrote {VEC} ({rows} vectors) and {PARAMS} (GEN_DEPTH={SHIFTREG['depth']})")


if __name__ == "__main__":
    main()

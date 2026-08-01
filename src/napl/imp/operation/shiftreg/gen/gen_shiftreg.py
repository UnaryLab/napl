"""
Generate golden test vectors for the shiftreg RTL module straight from napl's
functional Python model (napl.sim.operation.shiftreg) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

shiftreg is stateful: a depth-DEPTH bit-serial delay line whose reset() state is
the alternating pattern reg[i] = i % 2 (NOT all-zeros). We drive the model from
reset() with a known input stream and record (input, output) every cycle. A
mid-stream reset checks recovery from changed state.

Output: ../vec/shiftreg.vec, one line per cycle:

    <in> <out>        (each 0/1, space-separated)
    R                 (pulse i_rst_n low here)

The sizing param DEPTH is the single source of truth here: it is read from the
op config (mirroring test_shiftreg.py's shiftreg_config), used to build the
model, AND emitted into ../vec/shiftreg_params.vh as `GEN_DEPTH so the testbench
overrides the RTL parameter with the same value. RTL and sim therefore inherit
DEPTH from one place; they cannot drift. The header also records pp_delay.

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

# Sizing and encoder settings mirror test_shiftreg.py.
SHIFTREG = {"depth": 2}
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def main():
    model = shiftreg(config=SHIFTREG)
    model.reset()

    stream = []
    for v in rep_values(CODEC["polarity"]):
        stream += encode_value(CODEC, v)
    reset_at = len(stream) // 2

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(
        f"`define GEN_DEPTH {SHIFTREG['depth']}\n"
        f"`define GEN_PP_DELAY {model.hw.pp_delay}\n"
    )

    rows = 0
    with VEC.open("w") as f:
        for idx, bit in enumerate(stream):
            if idx == reset_at:
                model.reset()
                f.write("R\n")
            inp = torch.tensor(bit, dtype=model.stype)
            out = int(model(inp).item())
            f.write(f"{bit} {out}\n")
            rows += 1
    print(
        f"wrote {VEC} ({rows} vectors, reset@{reset_at}) and {PARAMS} "
        f"(GEN_DEPTH={SHIFTREG['depth']}, GEN_PP_DELAY={model.hw.pp_delay})"
    )


if __name__ == "__main__":
    main()

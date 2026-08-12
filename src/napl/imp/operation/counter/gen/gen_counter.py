"""
Generate golden test vectors for the counter RTL straight from napl's functional
Python model (napl.sim.operation.counter) -- so the testbench checks the Verilog
against the *actual* simulator, not a hand-derived truth table.

counter is a stateful, unipolar-only per-timestep streaming op: a saturating
running-count register whose reset() state is 0. We drive the model from reset()
with the test's encoder streams and record (input, output) every cycle. A
mid-stream reset checks recovery from a saturated state.

Output: ../vec/counter.vec, one line per cycle:

    <rst> <i_input> <o_output>     (each 0/1, space-separated)

`rst`=1 marks a cycle where the model was reset() BEFORE this timestep (the
testbench pulses i_rst_n low on that cycle).

The sizing param WIDTH is the single source of truth here: it is read from the
op config (mirroring test_counter.py's _WIDTH), used to build the model, AND
emitted into ../vec/counter_params.vh as `GEN_WIDTH so the testbench overrides the
RTL parameter with the same value. RTL and sim therefore inherit WIDTH from one
place; they cannot drift.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_counter.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "counter.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "counter_params.vh"

# Sizing and encoder settings mirror test_counter.py (unipolar only).
COUNTER = {"width": 3}
CODEC = {"polarity": "unipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def rep_stream():
    """The test's encoder streams for representative unipolar operands, concatenated."""
    stream = []
    for v in rep_values(CODEC["polarity"]):
        stream += encode_value(CODEC, v)
    return stream


def main():
    model = counter(config=COUNTER)
    model.reset()

    VEC.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(f"`define GEN_WIDTH {COUNTER['width']}\n")

    rows = 0
    with VEC.open("w") as f:
        # rst marks the first cycle after reset().
        first = True
        for b in rep_stream():
            out = int(model(torch.tensor(b, dtype=model.stype)).item())
            f.write(f"{int(first)} {b} {out}\n")
            first = False
            rows += 1

        # Saturate the count before testing a mid-stream reset.
        for _ in range(16):
            out = int(model(torch.tensor(1, dtype=model.stype)).item())
            f.write(f"0 1 {out}\n")
            rows += 1

        model.reset()
        first = True
        for b in rep_stream():
            out = int(model(torch.tensor(b, dtype=model.stype)).item())
            f.write(f"{int(first)} {b} {out}\n")
            first = False
            rows += 1

    print(f"wrote {VEC} ({rows} vectors) and {PARAMS} (GEN_WIDTH={COUNTER['width']})")


if __name__ == "__main__":
    main()

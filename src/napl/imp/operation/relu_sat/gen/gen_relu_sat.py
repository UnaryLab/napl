"""
Generate golden test vectors for the relu_sat RTL straight from napl's functional
Python model (napl.sim.operation.relu_sat) -- so the testbench checks the Verilog
against the *actual* simulator, not a hand-derived truth table.

relu_sat is a stateful per-timestep streaming op (two chained add_scale
accumulators) and is bipolar/rate-coded only, so there is a single variant
(no polarity split). It carries NO config-derived sizing key: its
super().__init__(config, [], ...) key_list is empty, and the inner add_scale
widths/scales are intrinsic algorithm constants baked into the Python model, not
config sizes. So there is NO sizing parameter to inherit; ../vec/relu_sat_params.vh
carries only the generated vector count.

Output: ../vec/relu_sat.vec, one line per cycle:

    <rst> <i_input> <o_out>      (each 0/1, space-separated)

`rst`=1 marks a cycle where the model was reset() BEFORE this timestep (the RTL
testbench pulses i_rst_n low on that cycle). We exercise:
  * the post-reset() stream for representative operands (saturation corners +
    in-range sweep + draws), driving the accumulators across their clamp rails;
  * a MID-STREAM reset from a deliberately dirtied accumulator state, proving the
    RTL's async i_rst_n returns to the exact post-reset() behavior the model has;
  * a short post-reset block that drives the add accumulator to its top state.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_relu_sat.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import relu_sat

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "relu_sat.vec"
PARAMS = Path(__file__).resolve().parent.parent / "vec" / "relu_sat_params.vh"

# Encoder settings mirror test_relu_sat.py.
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def rep_stream():
    """The test's encoder streams for representative operands, concatenated."""
    stream = []
    for v in rep_values(CODEC["polarity"]):
        stream += encode_value(CODEC, v)
    return stream


def main():
    model = relu_sat()
    model.reset()

    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with VEC.open("w") as f:
        # rst marks the first cycle after reset().
        first = True
        for b in rep_stream():
            out = int(model(torch.tensor(b, dtype=model.stype)).item())
            f.write(f"{int(first)} {b} {out}\n")
            first = False
            rows += 1

        # Saturate the accumulators before testing a mid-stream reset.
        for _ in range(16):
            b = 1
            out = int(model(torch.tensor(b, dtype=model.stype)).item())
            f.write(f"0 {b} {out}\n")
            rows += 1

        model.reset()
        first = True
        for b in rep_stream():
            out = int(model(torch.tensor(b, dtype=model.stype)).item())
            f.write(f"{int(first)} {b} {out}\n")
            first = False
            rows += 1

        # From reset, two ones fire the sub stage while the add accumulator holds
        # one unit, which drives it to its top state of 2 half-units; the ones and
        # zeros after it read that state back through the output.
        model.reset()
        first = True
        peak = 0
        for b in [1, 1, 0, 0, 1, 1, 1, 0]:
            out = int(model(torch.tensor(b, dtype=model.stype)).item())
            peak = max(peak, int(2 * model.add_1.accumulator.max().item()))
            f.write(f"{int(first)} {b} {out}\n")
            first = False
            rows += 1
        assert peak == 2, f"add accumulator reached {peak} half-units, not 2"

    PARAMS.write_text(f"`define GEN_VECTORS {rows}\n")

    print(f"wrote {VEC} ({rows} vectors)")


if __name__ == "__main__":
    main()

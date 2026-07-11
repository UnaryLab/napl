"""
Generate golden test vectors for the relu_sat RTL straight from napl's functional
Python model (napl.operation.relu_sat) -- so the testbench checks the Verilog
against the *actual* simulator, not a hand-derived truth table.

relu_sat is a stateful per-timestep streaming op (two chained add_any
accumulators) and is bipolar/rate-coded only, so there is a single variant
(no polarity split). It carries NO config-derived sizing key: its
super().__init__(config, [], ...) key_list is empty, and the inner add_any
widths/scales are intrinsic algorithm constants baked into the Python model, not
config sizes. So there is NO sizing parameter to inherit and NO *_params.vh is
emitted; the RTL is validate-only.

Output: ../vec/relu_sat.vec, one line per cycle:

    <rst> <i_in> <o_out>      (each 0/1, space-separated)

`rst`=1 marks a cycle where the model was reset() BEFORE this timestep (the RTL
testbench pulses i_rst_n low on that cycle). We exercise:
  * the post-reset() stream for representative operands (saturation corners +
    in-range sweep + draws), driving the accumulators across their clamp rails;
  * a MID-STREAM reset from a deliberately dirtied accumulator state, proving the
    RTL's async i_rst_n returns to the exact post-reset() behavior the model has.

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_relu_sat.py
"""
import sys
from pathlib import Path

import torch
from napl.operation import relu_sat

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "relu_sat.vec"

# test_relu_sat.py codec_config: the encoder feeding relu_sat.
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
        # the first row after reset() carries rst=1 (tb pulses i_rst_n low).
        first = True
        for b in rep_stream():
            out = int(model(torch.tensor(b, dtype=model.stype)).item())
            f.write(f"{int(first)} {b} {out}\n")
            first = False
            rows += 1

        # --- mid-stream reset equivalence ---------------------------------
        # Dirty the accumulators with a run of all-1s (drives them up to the
        # saturation rail), then reset() and replay a fresh stream. The first
        # post-reset row carries rst=1; the RTL's async i_rst_n must reproduce
        # the model's reset() so outputs match from a dirtied state onward.
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

    print(f"wrote {VEC} ({rows} vectors)")


if __name__ == "__main__":
    main()

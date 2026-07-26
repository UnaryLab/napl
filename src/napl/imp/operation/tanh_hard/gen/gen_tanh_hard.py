"""
Generate golden test vectors for the tanh_hard RTL straight from napl's
functional Python model (napl.sim.operation.tanh_hard) -- so the testbench checks
the Verilog against the *actual* simulator, not a hand-derived truth table.

tanh_hard is a stateless combinational identity pass-through (forward() ticks
the timestep counter and returns the input unchanged) and is polarity-agnostic,
so there is a single variant (no polarity split). A 1-bit input has only two
values, so an exhaustive sweep (2 rows) fully characterizes it; we also drive a
longer mixed stream so the testbench exercises many cycles.

Output: ../vec/tanh_hard.vec, one line per cycle:

    <i_input> <o_out>      (each 0/1, space-separated)

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_tanh_hard.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import tanh_hard

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values

VEC = Path(__file__).resolve().parent.parent / "vec" / "tanh_hard.vec"

# test_tanh_hard.py codec_config: the encoder feeding tanh_hard.
CODEC = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}


def build_stream():
    # exhaustive both 1-bit values (full characterization of this stateless op)
    # followed by the test's encoder streams for representative operands.
    stream = [0, 1, 1, 0]
    for v in rep_values(CODEC["polarity"]):
        stream += encode_value(CODEC, v)
    return stream


def main():
    model = tanh_hard()
    model.reset()

    stream = build_stream()
    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with VEC.open("w") as f:
        for b in stream:
            out = int(model(torch.tensor(b)).item())
            f.write(f"{b} {out}\n")
            rows += 1
    print(f"wrote {VEC} ({rows} vectors)")


if __name__ == "__main__":
    main()

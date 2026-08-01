"""
Generate golden test vectors for the lt_rc RTL module straight from napl's
functional Python model (napl.sim.operation.lt_rc) -- so the testbench checks the
Verilog against the *actual* simulator, not a hand-derived truth table.

lt_rc is stateful: it embeds a sync_skewed (width=2 -> a 0..3 saturating
counter, cnt) plus its own less-than dff. reset() zeroes both cnt and dff.
The output each cycle is the *registered* dff value (the state from the
previous cycle), so the op has one cycle of input->output latency.

Output: ../vec/lt_rc.vec, one line per timestep of every stream:

    <rst> <in_0> <in_1> <out>      (each 0/1, space-separated)

rst==1 marks the first cycle of a new reset segment: the model was reset()
there, so the testbench pulses i_rst_n low to reload cnt<=0, dff<=0 before
sampling that cycle. We replay several bit-streams (deterministic patterns that
exercise counter saturation at both ends and all 00/01/10/11 input
combinations), calling model.reset() before each stream and recording per-cycle
(inputs, output).

Run inside the `napl` conda env (so `import napl` resolves):
    python gen/gen_lt_rc.py
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import lt_rc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_pairs

VEC = Path(__file__).resolve().parent.parent / "vec" / "lt_rc.vec"

# Encoder settings mirror test_lt_rc.py and use distinct Sobol dimensions.
CODEC0 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 1}
CODEC1 = {"polarity": "bipolar", "timestep": 256, "generator": "sobol", "dim": 2}


def stream_pairs():
    """Yield one (in_0, in_1) bit-pair stream per reset segment.

    Each segment encodes one representative operand pair with the test's two
    encoders, so every segment is a test-derived stream; the per-segment reset
    also exercises mid-stream reset equivalence (cnt<=0, dff<=0)."""
    for v0, v1 in rep_pairs("bipolar", "bipolar", n=4):
        s0 = encode_value(CODEC0, v0)
        s1 = encode_value(CODEC1, v1)
        yield list(zip(s0, s1))


def main():
    VEC.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    segs = 0
    with VEC.open("w") as f:
        for pairs in stream_pairs():
            model = lt_rc(config={})
            model.reset()
            segs += 1
            for i, (a, b) in enumerate(pairs):
                in_0 = torch.tensor(a, dtype=torch.int8)
                in_1 = torch.tensor(b, dtype=torch.int8)
                out = int(model(in_0, in_1).item())
                rst = 1 if i == 0 else 0  # first cycle of a reset segment
                f.write(f"{rst} {a} {b} {out}\n")
                rows += 1
    print(f"wrote {VEC} ({rows} vectors, {segs} reset segments)")


if __name__ == "__main__":
    main()
